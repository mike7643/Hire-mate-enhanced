package com.hiremate.file;

import java.io.IOException;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.contentstream.operator.color.SetNonStrokingColor;
import org.apache.pdfbox.contentstream.operator.color.SetNonStrokingColorN;
import org.apache.pdfbox.contentstream.operator.color.SetNonStrokingColorSpace;
import org.apache.pdfbox.contentstream.operator.color.SetNonStrokingDeviceCMYKColor;
import org.apache.pdfbox.contentstream.operator.color.SetNonStrokingDeviceGrayColor;
import org.apache.pdfbox.contentstream.operator.color.SetNonStrokingDeviceRGBColor;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.graphics.PDXObject;
import org.apache.pdfbox.pdmodel.graphics.color.PDColor;
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;

/** 제출 PDF를 페이지별로 읽는다. 원문을 요약하거나 길이로 자르지 않는다. */
public final class PdfText {

    private PdfText() {
    }

    /** 기존 업로드 경로의 본문 추출. 읽지 못한 문서는 빈 문자열이다. */
    public static String extract(byte[] pdf) {
        return parse(pdf, false).text();
    }

    /** 페이지별 텍스트와 재검토 사유. 텍스트 추출 여부는 내용의 완전성을 보증하지 않는다. */
    public static Result parse(byte[] pdf) {
        return parse(pdf, true);
    }

    private static Result parse(byte[] pdf, boolean layout) {
        try (PDDocument document = Loader.loadPDF(pdf)) {
            if (!document.getCurrentAccessPermission().canExtractContent()) {
                return new Result(0, List.of(), List.of("TEXT_EXTRACTION_NOT_ALLOWED"));
            }
            List<Page> pages = new ArrayList<>();
            for (int number = 1; number <= document.getNumberOfPages(); number++) {
                pages.add(readPage(document, number, layout));
            }
            List<String> issues = pages.isEmpty() ? List.of("NO_PAGES") : List.of();
            return new Result(document.getNumberOfPages(), pages, issues);
        } catch (IOException | RuntimeException exception) {
            return new Result(0, List.of(), List.of("PDF_UNREADABLE_OR_PASSWORD_REQUIRED"));
        }
    }

    private static Page readPage(PDDocument document, int number, boolean layout) {
        try {
            PositionedTextStripper stripper = new PositionedTextStripper();
            stripper.setSortByPosition(true);
            stripper.setStartPage(number);
            stripper.setEndPage(number);
            String text = stripper.getText(document).strip();
            List<String> issues = new ArrayList<>();
            if (text.isBlank()) {
                issues.add("NO_TEXT_OCR_OR_BLANK_PAGE_REVIEW_REQUIRED");
            }
            if (text.indexOf(0xFFFD) >= 0 || text.indexOf('\0') >= 0) {
                issues.add("SUSPECT_TEXT_ENCODING");
            }
            PDPage page = document.getPage(number - 1);
            int images = 0;
            int forms = 0;
            if (page.getResources() != null && page.getResources().getXObjectNames().iterator().hasNext()) {
                issues.add("GRAPHICS_OR_IMAGES_REQUIRE_VISUAL_REVIEW");
                for (COSName name : page.getResources().getXObjectNames()) {
                    PDXObject object = page.getResources().getXObject(name);
                    if (object instanceof PDImageXObject) {
                        images++;
                    } else if (object instanceof PDFormXObject) {
                        forms++;
                    }
                }
            }
            var gridLines = layout && !stripper.spans.isEmpty() ? PdfGridLines.read(page) : List.<GridLine>of();
            return new Page(number, text, issues, stripper.spans, images, forms, gridLines);
        } catch (IOException | RuntimeException exception) {
            return new Page(number, "", List.of("PAGE_EXTRACTION_FAILED"));
        }
    }

    private static final class PositionedTextStripper extends PDFTextStripper {
        private final List<TextSpan> spans = new ArrayList<>();
        /** 글자를 그릴 때의 채우기 색. 정렬 뒤 writeString에서는 그리기 상태가 이미 지나가 있어 미리 남긴다. */
        private final Map<TextPosition, PDColor> colors = new IdentityHashMap<>();

        private PositionedTextStripper() throws IOException {
            addOperator(new SetNonStrokingColor(this));
            addOperator(new SetNonStrokingColorN(this));
            addOperator(new SetNonStrokingColorSpace(this));
            addOperator(new SetNonStrokingDeviceCMYKColor(this));
            addOperator(new SetNonStrokingDeviceGrayColor(this));
            addOperator(new SetNonStrokingDeviceRGBColor(this));
        }

        @Override
        public boolean isShouldProcessColorOperators() {
            return true;
        }

        @Override
        protected void processTextPosition(TextPosition text) {
            colors.put(text, getGraphicsState().getNonStrokingColor());
            super.processTextPosition(text);
        }

        @Override
        protected void writeString(String text, List<TextPosition> positions) throws IOException {
            super.writeString(text, positions);
            if (!positions.isEmpty() && !text.isBlank()) {
                TextPosition first = positions.getFirst();
                TextPosition last = positions.getLast();
                spans.add(new TextSpan(text, first.getXDirAdj(), first.getYDirAdj(), last.getEndX(),
                        mostCommon(positions, PositionedTextStripper::pointSize),
                        mostCommon(positions, position -> fillColor(colors.get(position)))));
            }
        }

        private static Float pointSize(TextPosition position) {
            // getFontSizeInPt()는 PDFBox에서 정수로 잘린다. 렌더링 행렬의 세로 배율을 쓴다.
            float size = position.getYScale();
            return Float.isFinite(size) && size > 0 ? Math.round(size * 100) / 100f : null;
        }

        private final Map<PDColor, String> rgbColors = new IdentityHashMap<>();

        private String fillColor(PDColor color) {
            if (color == null) {
                return null;
            }
            if (!rgbColors.containsKey(color)) {
                rgbColors.put(color, toHex(color));
            }
            return rgbColors.get(color);
        }

        private static String toHex(PDColor color) {
            try {
                return String.format("#%06x", color.toRGB() & 0xFFFFFF);
            } catch (IOException | RuntimeException exception) {
                return null;
            }
        }

        /** 조각 안에서 글자 수가 가장 많은 값. 크기나 색이 섞인 조각은 대부분의 글자 모양을 따른다. */
        private static <T> T mostCommon(List<TextPosition> positions, Function<TextPosition, T> value) {
            Map<T, Integer> counts = new LinkedHashMap<>();
            for (TextPosition position : positions) {
                T item = value.apply(position);
                if (item != null) {
                    counts.merge(item, 1, Integer::sum);
                }
            }
            return counts.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
        }
    }

    /**
     * 글자 조각과 위치, 모양. size는 pt, color는 채우기 색(#rrggbb)이고 읽지 못하면 null이다.
     * 모양은 제목과 본문을 가르는 단서일 뿐이고, 흰 글자처럼 보이지 않는 글도 그대로 넘긴다.
     */
    public record TextSpan(String text, float x, float y, float endX, Float size, String color) {
    }

    public record GridLine(float x1, float y1, float x2, float y2) {
    }

    public record Page(int number, String text, List<String> issues, List<TextSpan> spans,
                       int imageCount, int formCount, List<GridLine> gridLines) {
        public Page {
            issues = List.copyOf(issues);
            spans = spans == null ? List.of() : List.copyOf(spans);
            gridLines = gridLines == null ? List.of() : List.copyOf(gridLines);
        }

        public Page(int number, String text, List<String> issues) {
            this(number, text, issues, List.of(), 0, 0, List.of());
        }
    }

    public record Result(int pageCount, List<Page> pages, List<String> issues) {
        public Result {
            pages = List.copyOf(pages);
            issues = List.copyOf(issues);
        }

        public String text() {
            return pages.stream().map(Page::text).collect(Collectors.joining("\n\n")).strip();
        }

        public boolean reviewRequired() {
            return !issues.isEmpty() || pages.stream().anyMatch(page -> !page.issues().isEmpty());
        }
    }
}

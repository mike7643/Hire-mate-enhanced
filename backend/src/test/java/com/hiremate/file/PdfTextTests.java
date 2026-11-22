package com.hiremate.file;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.PDResources;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject;
import org.apache.pdfbox.pdmodel.graphics.image.LosslessFactory;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;
import org.apache.pdfbox.util.Matrix;
import org.junit.jupiter.api.Test;

class PdfTextTests {

    @Test
    void preservesTextBeyondOldLimitAndPageNumbers() throws Exception {
        try (PDDocument document = new PDDocument()) {
            for (int number = 0; number < 120; number++) {
                PDPage page = new PDPage();
                document.addPage(page);
                try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                    content.beginText();
                    content.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 8);
                    content.newLineAtOffset(20, 700);
                    content.showText("Experience ".repeat(50) + "Page " + number);
                    content.endText();
                }
            }
            byte[] bytes = save(document);
            PdfText.Result result = PdfText.parse(bytes);
            assertThat(result.pageCount()).isEqualTo(120);
            assertThat(result.pages()).hasSize(120);
            assertThat(result.pages().getLast().number()).isEqualTo(120);
            assertThat(result.text()).hasSizeGreaterThan(50_000).contains("Page 119");
            assertThat(PdfText.extract(bytes)).isEqualTo(result.text());
            assertThat(result.reviewRequired()).isFalse();
        }
    }

    @Test
    void extractsPositionedTextAsLayoutHints() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.beginText();
                content.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
                content.newLineAtOffset(40, 700);
                content.showText("Project experience");
                content.endText();
            }
            PdfText.Page parsed = PdfText.parse(save(document)).pages().getFirst();
            assertThat(parsed.spans()).isNotEmpty();
            assertThat(parsed.spans().stream().map(PdfText.TextSpan::text))
                    .anyMatch(text -> text.contains("Project"));
            assertThat(parsed.spans()).allSatisfy(span -> {
                assertThat(span.x()).isGreaterThanOrEqualTo(0);
                assertThat(span.endX()).isGreaterThan(span.x());
            });
            assertThat(parsed.imageCount()).isZero();
            assertThat(parsed.formCount()).isZero();
        }
    }

    @Test
    void extractsPaintedGridRulesInTextCoordinates() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.beginText();
                content.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
                content.newLineAtOffset(50, 680);
                content.showText("Cell text");
                content.endText();
                content.moveTo(40, 700);
                content.lineTo(200, 700);
                content.stroke();
                content.addRect(40, 650, 0.5f, 50);
                content.fill();
                content.addRect(40, 650, 160, 50);
                content.fill();
                content.setLineDashPattern(new float[] {2, 2}, 0);
                content.moveTo(150, 700);
                content.lineTo(150, 650);
                content.stroke();
            }
            assertThat(PdfText.parse(save(document)).pages().getFirst().gridLines()).containsExactly(
                    new PdfText.GridLine(40, 92, 200, 92), new PdfText.GridLine(40.25f, 92, 40.25f, 142));
        }
    }

    @Test
    void recordsFontSizeAndFillColorPerSpan() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                PDType1Font font = new PDType1Font(Standard14Fonts.FontName.HELVETICA);
                content.beginText();
                content.setFont(font, 11);
                content.setNonStrokingColor(new Color(0xE1, 0x57, 0x59));
                content.newLineAtOffset(40, 700);
                content.showText("Headline");
                content.setFont(font, 9.6f);
                content.setNonStrokingColor(new Color(0x33, 0x33, 0x33));
                content.newLineAtOffset(0, -30);
                content.showText("Body text");
                content.setFont(font, 0.8f);
                content.setNonStrokingColor(Color.WHITE);
                content.newLineAtOffset(0, -30);
                content.showText("Hidden instruction");
                content.endText();
            }
            var spans = PdfText.parse(save(document)).pages().getFirst().spans();
            assertThat(spans).extracting(PdfText.TextSpan::text, PdfText.TextSpan::size, PdfText.TextSpan::color)
                    .containsExactly(
                            tuple("Headline", 11f, "#e15759"),
                            tuple("Body text", 9.6f, "#333333"),
                            tuple("Hidden instruction", 0.8f, "#ffffff"));
        }
    }

    @Test
    void layoutEndMatchesExperimentOnRotatedPages() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            page.setRotation(90);
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.beginText();
                content.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
                content.setTextMatrix(Matrix.getRotateInstance(Math.PI / 2, 40, 700));
                content.showText("Project experience");
                content.endText();
            }
            var expected = new ArrayList<Float>();
            var stripper = new PDFTextStripper() {
                @Override
                protected void writeString(String text, List<TextPosition> positions) throws IOException {
                    if (!positions.isEmpty() && !text.isBlank()) {
                        expected.add(positions.getLast().getEndX());
                    }
                }
            };
            stripper.setSortByPosition(true);
            stripper.getText(document);
            var actual = PdfText.parse(save(document)).pages().getFirst().spans();
            assertThat(actual).extracting(PdfText.TextSpan::endX).containsExactlyElementsOf(expected);
        }
    }

    @Test
    void distinguishesImageAndFormResources() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            document.addPage(page);
            PDResources resources = new PDResources();
            resources.put(COSName.getPDFName("Image1"), LosslessFactory.createFromImage(document,
                    new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB)));
            resources.put(COSName.getPDFName("Form1"), new PDFormXObject(document));
            page.setResources(resources);
            PdfText.Page parsed = PdfText.parse(save(document)).pages().getFirst();
            assertThat(parsed.imageCount()).isEqualTo(1);
            assertThat(parsed.formCount()).isEqualTo(1);
        }
    }

    @Test
    void flagsBlankPageWithoutLosingOtherPages() throws Exception {
        try (PDDocument document = new PDDocument()) {
            document.addPage(new PDPage());
            PdfText.Result result = PdfText.parse(save(document));
            assertThat(result.pageCount()).isEqualTo(1);
            assertThat(result.pages().getFirst().issues())
                    .contains("NO_TEXT_OCR_OR_BLANK_PAGE_REVIEW_REQUIRED");
            assertThat(result.reviewRequired()).isTrue();
        }
    }

    @Test
    void reportsUnreadableInputAndKeepsLegacyEmptyText() {
        PdfText.Result result = PdfText.parse(new byte[] {1, 2, 3});
        assertThat(result.issues()).contains("PDF_UNREADABLE_OR_PASSWORD_REQUIRED");
        assertThat(result.reviewRequired()).isTrue();
        assertThat(PdfText.extract(new byte[] {1, 2, 3})).isEmpty();
    }

    private static byte[] save(PDDocument document) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        document.save(output);
        return output.toByteArray();
    }
}

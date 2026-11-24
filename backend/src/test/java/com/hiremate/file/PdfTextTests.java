package com.hiremate.file;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.pdfbox.cos.COSDictionary;
import org.apache.pdfbox.cos.COSInteger;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDFormContentStream;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.PDResources;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject;
import org.apache.pdfbox.pdmodel.graphics.form.PDTransparencyGroup;
import org.apache.pdfbox.pdmodel.graphics.image.LosslessFactory;
import org.apache.pdfbox.pdmodel.graphics.state.PDExtendedGraphicsState;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;
import org.apache.pdfbox.util.Matrix;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;

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
    void emitsOnlyPaintedFillRectanglesInTheTextCoordinateSystem() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle(240, 160));
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.setNonStrokingColor(0.8f);
                content.addRect(20, 120, 150, 20);
                content.fill();
                content.setNonStrokingColor(new Color(0x24, 0x68, 0xAC));
                content.setLineWidth(4);
                content.addRect(20, 96, 36, 18);
                content.fillAndStroke();
                content.setNonStrokingColor(Color.BLACK);
                content.addRect(20, 80, 0.75f, 0.5f);
                content.addRect(40, 70, 110, 0.5f);
                content.fill();
                content.addRect(20, 50, 30, 12);
                content.stroke();
                content.saveGraphicsState();
                content.addRect(60, 50, 30, 12);
                content.clip();
                content.restoreGraphicsState();
                writeText(content, 26, 128, "Header");
            }
            var parsed = PdfText.parse(save(document)).pages().getFirst();
            var layout = new ObjectMapper().valueToTree(parsed).path("layout");
            assertThat(layout.isObject()).as("painted layout evidence").isTrue();
            assertThat(layout.path("width").floatValue()).isEqualTo(240f);
            assertThat(layout.path("height").floatValue()).isEqualTo(160f);
            assertThat(layout.path("imageBoxes").isArray()).isTrue();
            assertThat(layout.path("imageBoxes").size()).isZero();
            assertThat(layout.path("fillRects").values()).extracting(rect -> rect.path("x1").floatValue(),
                    rect -> rect.path("y1").floatValue(), rect -> rect.path("x2").floatValue(),
                    rect -> rect.path("y2").floatValue(), rect -> rect.path("color").asString())
                    .containsExactly(tuple(20f, 20f, 170f, 40f, "#cccccc"),
                            tuple(20f, 46f, 56f, 64f, "#2468ac"),
                            tuple(20f, 79.5f, 20.75f, 80f, "#000000"),
                            tuple(40f, 89.5f, 150f, 90f, "#000000"));
            assertThat(parsed.spans()).extracting(PdfText.TextSpan::text, PdfText.TextSpan::x, PdfText.TextSpan::y)
                    .containsExactly(tuple("Header", 26f, 32f));
            assertThat(parsed.gridLines()).containsExactly(new PdfText.GridLine(40, 89.75f, 150, 89.75f));
        }
    }

    @Test
    void recordsSmallDrawnMarksSuchAsListBulletsWithoutTreatingLargeShapesAsMarks() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle(240, 160));
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.setNonStrokingColor(Color.BLACK);
                circle(content, 22, 140, 2);
                content.fill();
                circle(content, 22, 120, 2);
                content.stroke();
                content.moveTo(20, 96);
                content.lineTo(25, 98.5f);
                content.lineTo(20, 101);
                content.closePath();
                content.fill();
                circle(content, 120, 80, 15);
                content.fill();
                content.addRect(20, 60, 4, 4);
                content.fill();
                content.saveGraphicsState();
                content.addRect(100, 0, 10, 10);
                content.clip();
                circle(content, 22, 40, 2);
                content.fill();
                content.restoreGraphicsState();
                writeText(content, 30, 138, "Bullet item");
            }
            var parsed = PdfText.parse(save(document)).pages().getFirst();
            assertThat(parsed.layout().markBoxes()).extracting(PdfText.ImageBox::x1, PdfText.ImageBox::y1,
                    PdfText.ImageBox::x2, PdfText.ImageBox::y2)
                    .containsExactly(tuple(20f, 18f, 24f, 22f), tuple(20f, 38f, 24f, 42f),
                            tuple(20f, 59f, 25f, 64f));
            assertThat(parsed.layout().fillRects()).extracting(PdfText.FillRect::x1, PdfText.FillRect::y1)
                    .containsExactly(tuple(20f, 96f));
            var json = new ObjectMapper().valueToTree(parsed).path("layout").path("markBoxes");
            assertThat(json.isArray()).isTrue();
            assertThat(json.size()).isEqualTo(3);
        }
    }

    private static void circle(PDPageContentStream content, float cx, float cy, float r) throws IOException {
        float k = 0.5523f * r;
        content.moveTo(cx + r, cy);
        content.curveTo(cx + r, cy + k, cx + k, cy + r, cx, cy + r);
        content.curveTo(cx - k, cy + r, cx - r, cy + k, cx - r, cy);
        content.curveTo(cx - r, cy - k, cx - k, cy - r, cx, cy - r);
        content.curveTo(cx + k, cy - r, cx + r, cy - k, cx + r, cy);
        content.closePath();
    }

    @Test
    void doesNotReportFillBoxesForCancelledPathsOrHoles() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle(240, 160));
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.addRect(20, 70, 40, 40);
                content.addRect(30, 80, 10, 10);
                content.addRect(80, 70, 20, 20);
                content.fillEvenOdd();
                content.addRect(120, 70, 20, 20);
                content.addRect(140, 70, -20, 20);
                content.fill();
            }
            var layout = PdfText.parse(save(document)).pages().getFirst().layout();
            assertThat(layout.fillRects()).containsExactly(new PdfText.FillRect(80, 70, 100, 90, "#000000"));
        }
    }

    @Test
    void recognizesRectangularLinePathsWithoutBoxingPolygons() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle(240, 160));
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.moveTo(10, 60);
                content.lineTo(70, 60);
                content.lineTo(70, 80);
                content.lineTo(10, 80);
                content.fill();
                content.moveTo(80, 70);
                content.lineTo(90, 85);
                content.lineTo(100, 70);
                content.fill();
                content.saveGraphicsState();
                content.transform(new Matrix(1, 0.5f, 0, 1, 0, 0));
                content.addRect(20, 60, 50, 10);
                content.fill();
                content.restoreGraphicsState();
            }
            var layout = PdfText.parse(save(document)).pages().getFirst().layout();
            assertThat(layout.fillRects()).containsExactly(new PdfText.FillRect(10, 80, 70, 100, "#000000"));
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
            var parsed = PdfText.parse(save(document)).pages().getFirst();
            assertThat(parsed.spans()).extracting(PdfText.TextSpan::endX).containsExactlyElementsOf(expected);
            assertThat(new ObjectMapper().valueToTree(parsed).has("layout")).isFalse();
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
    void recordsImagePlacementsRatherThanResourceEntries() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle(240, 160));
            document.addPage(page);
            var drawn = LosslessFactory.createFromImage(document, new BufferedImage(2, 3, BufferedImage.TYPE_INT_RGB));
            var unused = LosslessFactory.createFromImage(document, new BufferedImage(4, 5, BufferedImage.TYPE_INT_RGB));
            var resources = new PDResources();
            resources.put(COSName.getPDFName("Unused"), unused);
            resources.put(COSName.getPDFName("UnusedForm"), new PDFormXObject(document));
            page.setResources(resources);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.drawImage(drawn, 40, 100, 60, 30);
                content.drawImage(drawn, new Matrix(0, 20, -10, 0, 150, 70));
            }
            var parsed = PdfText.parse(save(document)).pages().getFirst();
            var layout = new ObjectMapper().valueToTree(parsed).path("layout");
            assertThat(layout.path("imageBoxes").values()).extracting(box -> box.path("x1").floatValue(),
                    box -> box.path("y1").floatValue(), box -> box.path("x2").floatValue(),
                    box -> box.path("y2").floatValue(), box -> box.size())
                    .containsExactly(tuple(40f, 30f, 100f, 60f, 4), tuple(140f, 70f, 150f, 90f, 4));
            assertThat(parsed.imageCount()).isEqualTo(2);
            assertThat(parsed.formCount()).isEqualTo(1);
            assertThat(parsed.text()).isEmpty();
            assertThat(parsed.spans()).isEmpty();
            assertThat(parsed.gridLines()).isEmpty();
        }
    }

    @Test
    void clipsLayoutToVisiblePageRegionsAndSkipsZeroAlphaPaint() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            page.setCropBox(new PDRectangle(240, 160));
            document.addPage(page);
            var image = LosslessFactory.createFromImage(document, new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB));
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.addRect(-5, 150, 20, 30);
                content.fill();
                content.drawImage(image, -10, 145, 30, 30);
                content.saveGraphicsState();
                content.addRect(30, 30, 40, 40);
                content.clip();
                var partialAlpha = new PDExtendedGraphicsState();
                partialAlpha.setNonStrokingAlphaConstant(0.5f);
                content.setGraphicsStateParameters(partialAlpha);
                content.addRect(20, 20, 30, 50);
                content.fill();
                content.drawImage(image, 60, 60, 30, 20);
                content.addRect(75, 40, 10, 10);
                content.fill();
                content.drawImage(image, 75, 40, 10, 10);
                content.restoreGraphicsState();
                var zeroAlpha = new PDExtendedGraphicsState();
                zeroAlpha.setNonStrokingAlphaConstant(0f);
                content.setGraphicsStateParameters(zeroAlpha);
                content.addRect(100, 100, 10, 10);
                content.fill();
                content.drawImage(image, 100, 100, 10, 10);
            }
            var layout = PdfText.parse(save(document)).pages().getFirst().layout();
            assertThat(layout.fillRects()).containsExactly(new PdfText.FillRect(0, 0, 15, 10, "#000000"),
                    new PdfText.FillRect(30, 90, 50, 130, "#000000"));
            assertThat(layout.imageBoxes()).containsExactly(new PdfText.ImageBox(0, 0, 20, 15),
                    new PdfText.ImageBox(60, 90, 70, 100));
        }
    }

    @Test
    void respectsTheOpacityOfPaintedTransparencyGroups() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle(240, 160));
            document.addPage(page);
            var image = LosslessFactory.createFromImage(document, new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB));
            var group = new PDTransparencyGroup(document);
            group.setBBox(new PDRectangle(25, 30));
            group.setResources(new PDResources());
            var transparency = new COSDictionary();
            transparency.setItem(COSName.S, COSName.TRANSPARENCY);
            group.getCOSObject().setItem(COSName.GROUP, transparency);
            try (var formContent = new PDFormContentStream(group)) {
                formContent.addRect(5, 5, 10, 4);
                formContent.fill();
                formContent.drawImage(image, 10, 10, 30, 30);
            }
            try (var content = new PDPageContentStream(document, page)) {
                content.saveGraphicsState();
                content.transform(Matrix.getTranslateInstance(40, 60));
                content.drawForm(group);
                content.restoreGraphicsState();
                content.transform(Matrix.getTranslateInstance(140, 60));
                var alpha = new PDExtendedGraphicsState();
                alpha.setNonStrokingAlphaConstant(0f);
                content.setGraphicsStateParameters(alpha);
                content.drawForm(group);
            }
            var parsed = PdfText.parse(save(document)).pages().getFirst();
            assertThat(parsed.layout().imageBoxes()).containsExactly(new PdfText.ImageBox(50, 70, 65, 90));
            assertThat(parsed.layout().fillRects()).containsExactly(new PdfText.FillRect(45, 91, 55, 95, "#000000"));
            assertThat(parsed.imageCount()).isZero();
            assertThat(parsed.formCount()).isEqualTo(1);
        }
    }

    @Test
    void keepsTextWhenAnOptionalGraphicsResourceIsUnreadable() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            document.addPage(page);
            var resources = new PDResources();
            var objects = new COSDictionary();
            objects.setItem(COSName.getPDFName("Invalid"), COSInteger.ONE);
            resources.getCOSObject().setItem(COSName.XOBJECT, objects);
            resources.put(COSName.getPDFName("Image1"), LosslessFactory.createFromImage(document,
                    new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB)));
            resources.put(COSName.getPDFName("Form1"), new PDFormXObject(document));
            page.setResources(resources);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                writeText(content, 40, 700, "Readable page");
                content.addRect(40, 660, 60, 20);
                content.fill();
            }
            byte[] pdf = save(document);
            var parsed = PdfText.parse(pdf).pages().getFirst();
            assertThat(parsed.text()).isEqualTo("Readable page");
            assertThat(parsed.spans()).extracting(PdfText.TextSpan::text).containsExactly("Readable page");
            assertThat(PdfText.extract(pdf)).isEqualTo(parsed.text());
            assertThat(parsed.imageCount()).isEqualTo(1);
            assertThat(parsed.formCount()).isEqualTo(1);
            assertThat(parsed.issues()).containsExactly("GRAPHICS_OR_IMAGES_REQUIRE_VISUAL_REVIEW");
            assertThat(parsed.layout().fillRects()).hasSize(1);
            assertThat(parsed.layout().imageBoxes()).isEmpty();
        }
    }

    @Test
    void capsLayoutWithoutLosingItWhenTheLegacyGridLimitIsExceeded() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            document.addPage(page);
            var image = LosslessFactory.createFromImage(document, new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB));
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                writeText(content, 40, 750, "Dense page");
                for (int i = 0; i < 513; i++) {
                    content.moveTo(20, 720);
                    content.lineTo(180, 720);
                }
                content.stroke();
                for (int i = 0; i < 8200; i++) {
                    content.addRect(20 + i % 200, 80 + i / 200, 0.5f, 0.5f);
                }
                content.fill();
                for (int i = 0; i < 260; i++) {
                    content.drawImage(image, 260, 700, 20, 30);
                }
            }
            var parsed = PdfText.parse(save(document)).pages().getFirst();
            assertThat(parsed.text()).isEqualTo("Dense page");
            assertThat(parsed.gridLines()).isEmpty();
            assertThat(parsed.layout()).isNotNull();
            assertThat(parsed.layout().fillRects()).hasSize(8192);
            assertThat(parsed.layout().imageBoxes()).hasSize(256);
            assertThat(parsed.imageCount()).isEqualTo(1);
        }
    }

    @Test
    void omitsLayoutForOversizedPathsWithoutDroppingText() throws Exception {
        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage();
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.moveTo(20, 200);
                for (int i = 0; i < 65_536; i++) {
                    content.lineTo(20 + i % 2, 201);
                }
                content.stroke();
                writeText(content, 40, 700, "Text after oversized path");
            }
            var parsed = PdfText.parse(save(document)).pages().getFirst();
            assertThat(parsed.text()).isEqualTo("Text after oversized path");
            assertThat(parsed.spans()).extracting(PdfText.TextSpan::text).containsExactly("Text after oversized path");
            assertThat(new ObjectMapper().valueToTree(parsed).has("layout")).isFalse();
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

    private static void writeText(PDPageContentStream content, float x, float y, String text) throws IOException {
        content.beginText();
        content.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
        content.newLineAtOffset(x, y);
        content.showText(text);
        content.endText();
    }

    private static byte[] save(PDDocument document) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        document.save(output);
        return output.toByteArray();
    }
}

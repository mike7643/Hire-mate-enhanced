package com.hiremate.file;

import java.awt.geom.GeneralPath;
import java.awt.geom.PathIterator;
import java.awt.geom.Point2D;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.pdfbox.contentstream.PDFGraphicsStreamEngine;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.graphics.image.PDImage;

/** 실제로 그린 수평, 수직 격자선만 텍스트와 같은 좌표로 남긴다. */
final class PdfGridLines extends PDFGraphicsStreamEngine {
    private static final int MAX_LINES = 512;
    private static final float MIN_LENGTH = 6;
    private static final float MAX_THICKNESS = 2;
    private final GeneralPath path = new GeneralPath();
    private final List<PdfText.GridLine> lines = new ArrayList<>();
    private final List<PdfText.GridLine> filledRules = new ArrayList<>();
    private boolean rectanglesOnly = true;
    private Integer clipRule;

    private PdfGridLines(PDPage page) {
        super(page);
    }

    static List<PdfText.GridLine> read(PDPage page) {
        var box = page.getCropBox();
        if (page.getRotation() != 0 || box.getLowerLeftX() != 0 || box.getLowerLeftY() != 0) {
            return List.of();
        }
        var reader = new PdfGridLines(page);
        try {
            reader.processPage(page);
            return reader.lines;
        } catch (IOException | RuntimeException exception) {
            // 격자 단서는 선택 사항이라 읽기 실패가 본문 추출 실패가 되지 않는다.
            return List.of();
        }
    }

    @Override
    public void appendRectangle(Point2D a, Point2D b, Point2D c, Point2D d) {
        path.moveTo(a.getX(), a.getY());
        path.lineTo(b.getX(), b.getY());
        path.lineTo(c.getX(), c.getY());
        path.lineTo(d.getX(), d.getY());
        path.closePath();
        boolean aligned = (a.getX() == b.getX() && b.getY() == c.getY()
                && c.getX() == d.getX() && d.getY() == a.getY())
                || (a.getY() == b.getY() && b.getX() == c.getX()
                && c.getY() == d.getY() && d.getX() == a.getX());
        if (!aligned || filledRules.size() > MAX_LINES) {
            return;
        }
        double width = Math.abs(a.getX() - c.getX());
        double height = Math.abs(a.getY() - c.getY());
        double x = (a.getX() + c.getX()) / 2;
        double y = (a.getY() + c.getY()) / 2;
        if (width >= MIN_LENGTH && height > 0 && height <= MAX_THICKNESS) {
            filledRules.add(rule(x - width / 2, y, x + width / 2, y));
        } else if (height >= MIN_LENGTH && width > 0 && width <= MAX_THICKNESS) {
            filledRules.add(rule(x, y - height / 2, x, y + height / 2));
        }
    }

    @Override
    public void moveTo(float x, float y) {
        rectanglesOnly = false;
        path.moveTo(x, y);
    }

    @Override
    public void lineTo(float x, float y) {
        path.lineTo(x, y);
    }

    @Override
    public void curveTo(float x1, float y1, float x2, float y2, float x3, float y3) {
        path.curveTo(x1, y1, x2, y2, x3, y3);
    }

    @Override
    public Point2D getCurrentPoint() {
        return path.getCurrentPoint();
    }

    @Override
    public void closePath() {
        path.closePath();
    }

    @Override
    public void drawImage(PDImage image) {
        // 래스터에서 격자를 추측하지 않는다.
    }

    @Override
    public void shadingFill(COSName name) {
        // 그라데이션은 격자선이 아니다.
    }

    @Override
    public void clip(int windingRule) {
        clipRule = windingRule;
    }

    @Override
    public void endPath() throws IOException {
        if (lines.size() > MAX_LINES) {
            throw new IOException("Too many grid lines");
        }
        if (clipRule != null) {
            path.setWindingRule(clipRule);
            getGraphicsState().intersectClippingPath(path);
            clipRule = null;
        }
        path.reset();
        filledRules.clear();
        rectanglesOnly = true;
    }

    @Override
    public void strokePath() throws IOException {
        collectStroke();
        endPath();
    }

    @Override
    public void fillPath(int windingRule) throws IOException {
        collectFill();
        endPath();
    }

    @Override
    public void fillAndStrokePath(int windingRule) throws IOException {
        collectFill();
        collectStroke();
        endPath();
    }

    private void collectFill() {
        if (rectanglesOnly && getGraphicsState().getNonStrokeAlphaConstant() > 0) {
            filledRules.forEach(this::addVisibleRule);
        }
    }

    private void collectStroke() {
        if (getGraphicsState().getLineDashPattern().getDashArray().length != 0
                || getGraphicsState().getAlphaConstant() <= 0
                || transformWidth(getGraphicsState().getLineWidth()) > MAX_THICKNESS) {
            return;
        }
        double[] coordinates = new double[6];
        double x = 0;
        double y = 0;
        double startX = 0;
        double startY = 0;
        for (var cursor = path.getPathIterator(null); !cursor.isDone() && lines.size() <= MAX_LINES; cursor.next()) {
            switch (cursor.currentSegment(coordinates)) {
                case PathIterator.SEG_MOVETO -> {
                    startX = coordinates[0];
                    startY = coordinates[1];
                }
                case PathIterator.SEG_LINETO -> addStroke(x, y, coordinates[0], coordinates[1]);
                case PathIterator.SEG_CLOSE -> {
                    addStroke(x, y, startX, startY);
                    coordinates[0] = startX;
                    coordinates[1] = startY;
                }
                case PathIterator.SEG_CUBICTO -> {
                    coordinates[0] = coordinates[4];
                    coordinates[1] = coordinates[5];
                }
                default -> throw new IllegalStateException("Unsupported path segment");
            }
            x = coordinates[0];
            y = coordinates[1];
        }
    }

    private void addStroke(double x1, double y1, double x2, double y2) {
        if ((x1 == x2 && Math.abs(y2 - y1) >= MIN_LENGTH) || (y1 == y2 && Math.abs(x2 - x1) >= MIN_LENGTH)) {
            addVisibleRule(rule(x1, y1, x2, y2));
        }
    }

    private void addVisibleRule(PdfText.GridLine line) {
        var clip = getGraphicsState().getCurrentClippingPath();
        double height = getPage().getCropBox().getHeight();
        // 잘린 획을 완전한 표 구분선으로 간주하지 않는다.
        if (clip.contains(line.x1(), height - line.y1()) && clip.contains(line.x2(), height - line.y2())) {
            lines.add(line);
        }
    }

    private PdfText.GridLine rule(double x1, double y1, double x2, double y2) {
        float height = getPage().getCropBox().getHeight();
        return new PdfText.GridLine((float) Math.min(x1, x2), height - (float) Math.max(y1, y2),
                (float) Math.max(x1, x2), height - (float) Math.min(y1, y2));
    }
}

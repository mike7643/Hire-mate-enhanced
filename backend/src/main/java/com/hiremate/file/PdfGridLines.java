package com.hiremate.file;

import java.awt.Shape;
import java.awt.geom.Area;
import java.awt.geom.GeneralPath;
import java.awt.geom.PathIterator;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.pdfbox.contentstream.PDFGraphicsStreamEngine;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.font.PDType3Font;
import org.apache.pdfbox.pdmodel.graphics.form.PDTransparencyGroup;
import org.apache.pdfbox.pdmodel.graphics.image.PDImage;
import org.apache.pdfbox.util.Matrix;
import org.apache.pdfbox.util.Vector;

/** 실제로 칠한 페이지 단서를 텍스트와 같은 좌표로 남긴다. */
final class PdfGridLines extends PDFGraphicsStreamEngine {
    static final Geometry EMPTY = new Geometry(List.of(), null);
    private static final int MAX_LINES = 512;
    private static final int MAX_FILL_RECTS = 8192;
    private static final int MAX_IMAGE_BOXES = 256;
    private static final int MAX_MARK_BOXES = 2048;
    // 목록 머리표로 볼 수 있는 작은 그림의 크기 범위(pt)와 가로세로 비율 한도.
    private static final double MIN_MARK = 1;
    private static final double MAX_MARK = 8;
    private static final double MAX_MARK_RATIO = 2;
    private static final int MAX_PATH_SEGMENTS = 65_536;
    private static final int MAX_WINDING_SEGMENTS = 1024;
    private static final float MIN_LENGTH = 6;
    private static final float MAX_THICKNESS = 2;
    private final float pageHeight;
    private final GeneralPath path = new GeneralPath();
    private final List<PdfText.GridLine> lines = new ArrayList<>();
    private final List<PdfText.GridLine> filledRules = new ArrayList<>();
    private final List<Rectangle2D.Double> rectangles = new ArrayList<>();
    private final List<PdfText.FillRect> fillRects = new ArrayList<>();
    private final List<PdfText.ImageBox> imageBoxes = new ArrayList<>();
    private final List<PdfText.ImageBox> markBoxes = new ArrayList<>();
    private boolean collectGridLines;
    private boolean rectanglesOnly = true;
    private boolean clockwiseRectangle;
    private boolean counterclockwiseRectangle;
    private boolean groupVisible = true;
    private Integer clipRule;
    private int pathSegments;
    // Type3 글꼴은 글자를 그림 명령으로 그린다. 글자 안의 그림은 머리표가 아니다.
    private int type3Depth;

    private PdfGridLines(PDPage page, float pageHeight, boolean wantGridLines) {
        super(page);
        this.pageHeight = pageHeight;
        collectGridLines = wantGridLines;
    }

    static Geometry read(PDPage page, boolean wantGridLines) {
        try {
            var box = page.getCropBox();
            if (page.getRotation() != 0 || box.getLowerLeftX() != 0 || box.getLowerLeftY() != 0
                    || !validSize(box.getWidth()) || !validSize(box.getHeight())) {
                return EMPTY;
            }
            var reader = new PdfGridLines(page, box.getHeight(), wantGridLines);
            reader.processPage(page);
            return new Geometry(reader.lines,
                    new PdfText.Layout(box.getWidth(), box.getHeight(), reader.fillRects, reader.imageBoxes,
                            reader.markBoxes));
        } catch (IOException | RuntimeException exception) {
            // 그래픽 단서는 선택 사항이라 읽기 실패가 본문 추출 실패가 되지 않는다.
            return EMPTY;
        }
    }

    private static boolean validSize(float size) {
        return Float.isFinite(size) && size > 0;
    }

    private void checkPathBudget(int segments) {
        pathSegments += segments;
        if (pathSegments > MAX_PATH_SEGMENTS) {
            throw new IllegalStateException("PDF layout path is too complex");
        }
    }

    @Override
    public void appendRectangle(Point2D a, Point2D b, Point2D c, Point2D d) {
        checkPathBudget(5);
        path.moveTo(a.getX(), a.getY());
        path.lineTo(b.getX(), b.getY());
        path.lineTo(c.getX(), c.getY());
        path.lineTo(d.getX(), d.getY());
        path.closePath();
        double cross = (b.getX() - a.getX()) * (c.getY() - b.getY())
                - (b.getY() - a.getY()) * (c.getX() - b.getX());
        clockwiseRectangle |= cross < 0;
        counterclockwiseRectangle |= cross > 0;
        boolean aligned = (a.getX() == b.getX() && b.getY() == c.getY()
                && c.getX() == d.getX() && d.getY() == a.getY())
                || (a.getY() == b.getY() && b.getX() == c.getX()
                && c.getY() == d.getY() && d.getX() == a.getX());
        if (!aligned) {
            return;
        }
        double width = Math.abs(a.getX() - c.getX());
        double height = Math.abs(a.getY() - c.getY());
        if (fillRects.size() < MAX_FILL_RECTS && width > 0 && height > 0) {
            rectangles.add(new Rectangle2D.Double(Math.min(a.getX(), c.getX()),
                    Math.min(a.getY(), c.getY()), width, height));
        }
        if (!collectGridLines || filledRules.size() > MAX_LINES) {
            return;
        }
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
        checkPathBudget(1);
        rectanglesOnly = false;
        path.moveTo(x, y);
    }

    @Override
    public void lineTo(float x, float y) {
        checkPathBudget(1);
        path.lineTo(x, y);
    }

    @Override
    public void curveTo(float x1, float y1, float x2, float y2, float x3, float y3) {
        checkPathBudget(1);
        path.curveTo(x1, y1, x2, y2, x3, y3);
    }

    @Override
    public Point2D getCurrentPoint() {
        return path.getCurrentPoint();
    }

    @Override
    public void closePath() {
        checkPathBudget(1);
        path.closePath();
    }

    @Override
    public void drawImage(PDImage image) {
        if (imageBoxes.size() >= MAX_IMAGE_BOXES || !visiblePaint()) {
            return;
        }
        // 이미지 자원은 읽지 않고 단위 정사각형이 실제로 놓인 좌표만 읽는다.
        var transform = getGraphicsState().getCurrentTransformationMatrix().createAffineTransform();
        var shape = transform.createTransformedShape(new Rectangle2D.Double(0, 0, 1, 1));
        var box = clippedBox(shape, false);
        if (box != null) {
            imageBoxes.add(box);
        }
    }

    private boolean visiblePaint() {
        var state = getGraphicsState();
        double alpha = state.getNonStrokeAlphaConstant();
        return groupVisible && Double.isFinite(alpha) && alpha > 0 && state.getSoftMask() == null;
    }

    @Override
    protected void showType3Glyph(Matrix textRenderingMatrix, PDType3Font font, int code, Vector displacement)
            throws IOException {
        type3Depth++;
        try {
            super.showType3Glyph(textRenderingMatrix, font, code, displacement);
        } finally {
            type3Depth--;
        }
    }

    @Override
    public void showTransparencyGroup(PDTransparencyGroup group) throws IOException {
        boolean parentVisible = groupVisible;
        // PDFBox는 그룹 내부의 알파와 마스크를 초기화하므로 그룹 자체의 가시성을 별도로 유지한다.
        groupVisible = visiblePaint();
        try {
            super.showTransparencyGroup(group);
        } finally {
            groupVisible = parentVisible;
        }
    }

    private PdfText.ImageBox clippedBox(Shape shape, boolean requireRectangle) {
        var bounds = shape.getBounds2D();
        var clip = getGraphicsState().getCurrentClippingPath();
        var clipBounds = clip.getBounds2D();
        if (bounds.isEmpty() || !Double.isFinite(bounds.getMaxX()) || !Double.isFinite(bounds.getMaxY())
                || !Double.isFinite(bounds.getMinX()) || !Double.isFinite(bounds.getMinY())
                || !clipBounds.intersects(bounds)) {
            return null;
        }
        if (shape instanceof Rectangle2D && clip.isRectangular()) {
            bounds = bounds.createIntersection(clipBounds);
        } else {
            var visible = new Area(shape);
            visible.intersect(clip);
            if (visible.isEmpty() || (requireRectangle && !visible.isRectangular())) {
                return null;
            }
            bounds = visible.getBounds2D();
        }
        float left = (float) bounds.getMinX();
        float right = (float) bounds.getMaxX();
        float top = (float) (pageHeight - bounds.getMaxY());
        float bottom = (float) (pageHeight - bounds.getMinY());
        return left < right && top < bottom ? new PdfText.ImageBox(left, top, right, bottom) : null;
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
    public void endPath() {
        if (clipRule != null) {
            path.setWindingRule(clipRule);
            getGraphicsState().intersectClippingPath(path);
            clipRule = null;
        }
        path.reset();
        pathSegments = 0;
        filledRules.clear();
        rectangles.clear();
        rectanglesOnly = true;
        clockwiseRectangle = false;
        counterclockwiseRectangle = false;
    }

    @Override
    public void strokePath() {
        collectMark(false);
        collectStroke();
        endPath();
    }

    @Override
    public void fillPath(int windingRule) {
        collectMark(true);
        collectFill(windingRule);
        endPath();
    }

    @Override
    public void fillAndStrokePath(int windingRule) {
        collectMark(true);
        collectFill(windingRule);
        collectStroke();
        endPath();
    }

    /** 사각형이 아닌 작은 닫힌 그림을 머리표 단서로 남긴다. 글자 그림, 큰 도형, 보이지 않는 그림은 넣지 않는다. */
    private void collectMark(boolean filled) {
        if (type3Depth > 0 || markBoxes.size() >= MAX_MARK_BOXES || rectanglesOnly || pathSegments < 3) {
            return;
        }
        var state = getGraphicsState();
        double alpha = filled ? state.getNonStrokeAlphaConstant() : state.getAlphaConstant();
        if (!groupVisible || !Double.isFinite(alpha) || alpha <= 0 || state.getSoftMask() != null) {
            return;
        }
        var bounds = path.getBounds2D();
        double width = bounds.getWidth();
        double height = bounds.getHeight();
        if (width < MIN_MARK || height < MIN_MARK || width > MAX_MARK || height > MAX_MARK
                || Math.max(width, height) / Math.min(width, height) > MAX_MARK_RATIO) {
            return;
        }
        var box = clippedBox(path, false);
        if (box != null) {
            markBoxes.add(box);
        }
    }

    private void collectFill(int windingRule) {
        if (rectanglesOnly && getGraphicsState().getNonStrokeAlphaConstant() > 0) {
            filledRules.forEach(this::addVisibleRule);
        }
        if (fillRects.size() < MAX_FILL_RECTS && visiblePaint()) {
            if (rectanglesOnly && !rectangles.isEmpty()) {
                collectFillRects(windingRule);
            } else if (!rectanglesOnly && pathSegments >= 4 && pathSegments <= 6) {
                // 짧은 선 경로도 실제로 사각형을 채웠다면 단서로 남긴다.
                var filled = filledArea(windingRule);
                if (filled.isRectangular() && !filled.isEmpty()) {
                    addFillRect(filled.getBounds2D(), PdfText.toHex(getGraphicsState().getNonStrokingColor()));
                }
            }
        }
    }

    private void collectFillRects(int windingRule) {
        String color = PdfText.toHex(getGraphicsState().getNonStrokingColor());
        if (color == null) {
            return;
        }
        // 방향이 다른 사각형이나 짝홀 채우기는 서로 지울 수 있다. 복잡한 경우 추측하지 않는다.
        boolean mayCancel = (clockwiseRectangle && counterclockwiseRectangle)
                || (windingRule == PathIterator.WIND_EVEN_ODD && pathSegments > 5);
        if (mayCancel && pathSegments > MAX_WINDING_SEGMENTS) {
            return;
        }
        Area filled = mayCancel ? filledArea(windingRule) : null;
        for (var rect : rectangles) {
            if (fillRects.size() >= MAX_FILL_RECTS) {
                break;
            }
            if (filled == null) {
                addFillRect(rect, color);
            } else {
                var section = new Area(rect);
                section.intersect(filled);
                addFillRect(section, color);
            }
        }
    }

    private Area filledArea(int windingRule) {
        path.setWindingRule(windingRule);
        return new Area(path);
    }

    private void addFillRect(Shape shape, String color) {
        if (color != null) {
            var box = clippedBox(shape, true);
            if (box != null) {
                fillRects.add(new PdfText.FillRect(box.x1(), box.y1(), box.x2(), box.y2(), color));
            }
        }
    }

    private void collectStroke() {
        if (!collectGridLines || getGraphicsState().getLineDashPattern().getDashArray().length != 0
                || getGraphicsState().getAlphaConstant() <= 0
                || transformWidth(getGraphicsState().getLineWidth()) > MAX_THICKNESS) {
            return;
        }
        double[] coordinates = new double[6];
        double x = 0;
        double y = 0;
        double startX = 0;
        double startY = 0;
        for (var cursor = path.getPathIterator(null); !cursor.isDone() && collectGridLines; cursor.next()) {
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
        if (!collectGridLines) {
            return;
        }
        var clip = getGraphicsState().getCurrentClippingPath();
        // 잘린 획을 완전한 표 구분선으로 간주하지 않는다.
        if (clip.contains(line.x1(), pageHeight - (double) line.y1())
                && clip.contains(line.x2(), pageHeight - (double) line.y2())) {
            if (lines.size() == MAX_LINES) {
                // 기존 정책대로 과밀 격자는 버리되 다른 배치 단서는 유지한다.
                lines.clear();
                collectGridLines = false;
            } else {
                lines.add(line);
            }
        }
    }

    private PdfText.GridLine rule(double x1, double y1, double x2, double y2) {
        return new PdfText.GridLine((float) Math.min(x1, x2), pageHeight - (float) Math.max(y1, y2),
                (float) Math.max(x1, x2), pageHeight - (float) Math.min(y1, y2));
    }

    record Geometry(List<PdfText.GridLine> gridLines, PdfText.Layout layout) {
    }
}

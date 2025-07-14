# Drawing Graphics and Canvas

## Table of Contents
- [Overview of Android Graphics](#overview-of-android-graphics)
- [Canvas and Paint Fundamentals](#canvas-and-paint-fundamentals)
- [Drawing Basic Shapes](#drawing-basic-shapes)
- [Path Drawing](#path-drawing)
- [Custom Views with Graphics](#custom-views-with-graphics)
- [Bitmap Operations](#bitmap-operations)
- [Matrix Transformations](#matrix-transformations)
- [Animation with Graphics](#animation-with-graphics)
- [Performance Optimization](#performance-optimization)
- [Advanced Drawing Techniques](#advanced-drawing-techniques)
- [Hardware Acceleration](#hardware-acceleration)
- [Best Practices](#best-practices)

## Overview of Android Graphics

Android provides a rich graphics system for creating custom visual elements and interactive user interfaces. The graphics system is built on several key components that work together to render content on the screen.

### Graphics System Components
- **Canvas**: The drawing surface for graphics operations
- **Paint**: Defines how to draw (colors, styles, effects)
- **Drawable**: Objects that can be drawn to a Canvas
- **Bitmap**: Pixel-based images
- **Path**: Complex shapes and curves
- **Matrix**: Geometric transformations

### Graphics APIs
- **2D Graphics**: Canvas, Paint, Path for 2D drawing
- **OpenGL ES**: Hardware-accelerated 3D graphics
- **Vulkan**: Low-level graphics API (Android 7.0+)
- **RenderScript**: Parallel computing platform

## Canvas and Paint Fundamentals

Canvas provides the interface for drawing, while Paint defines the drawing properties.

### Canvas Class
The Canvas class provides methods for drawing shapes, text, and images.

```java
public class BasicCanvasView extends View {
    private Paint paint;
    
    public BasicCanvasView(Context context, AttributeSet attrs) {
        super(context, attrs);
        initPaint();
    }
    
    private void initPaint() {
        paint = new Paint();
        paint.setAntiAlias(true);  // Smooth edges
        paint.setColor(Color.BLUE);
        paint.setStyle(Paint.Style.FILL);
        paint.setStrokeWidth(5f);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        // Canvas provides the drawing surface
        int width = getWidth();
        int height = getHeight();
        
        // Draw background
        canvas.drawColor(Color.WHITE);
        
        // Draw a simple circle
        float centerX = width / 2f;
        float centerY = height / 2f;
        float radius = Math.min(width, height) / 4f;
        
        canvas.drawCircle(centerX, centerY, radius, paint);
    }
}
```

### Paint Class Properties
Paint controls how drawing operations appear on the canvas.

```java
public class PaintPropertiesView extends View {
    private Paint[] paints;
    
    public PaintPropertiesView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setupPaints();
    }
    
    private void setupPaints() {
        paints = new Paint[6];
        
        // Solid fill paint
        paints[0] = new Paint();
        paints[0].setColor(Color.RED);
        paints[0].setStyle(Paint.Style.FILL);
        paints[0].setAntiAlias(true);
        
        // Stroke paint
        paints[1] = new Paint();
        paints[1].setColor(Color.GREEN);
        paints[1].setStyle(Paint.Style.STROKE);
        paints[1].setStrokeWidth(8f);
        paints[1].setAntiAlias(true);
        
        // Fill and stroke paint
        paints[2] = new Paint();
        paints[2].setColor(Color.BLUE);
        paints[2].setStyle(Paint.Style.FILL_AND_STROKE);
        paints[2].setStrokeWidth(4f);
        paints[2].setAntiAlias(true);
        
        // Gradient paint
        paints[3] = new Paint();
        paints[3].setShader(new LinearGradient(
            0, 0, 200, 200,
            Color.YELLOW, Color.MAGENTA,
            Shader.TileMode.CLAMP));
        paints[3].setAntiAlias(true);
        
        // Shadow paint
        paints[4] = new Paint();
        paints[4].setColor(Color.CYAN);
        paints[4].setShadowLayer(10f, 5f, 5f, Color.GRAY);
        paints[4].setAntiAlias(true);
        
        // Text paint
        paints[5] = new Paint();
        paints[5].setColor(Color.BLACK);
        paints[5].setTextSize(48f);
        paints[5].setTypeface(Typeface.DEFAULT_BOLD);
        paints[5].setAntiAlias(true);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        int spacing = 120;
        int startY = 100;
        
        // Draw circles with different paint styles
        for (int i = 0; i < 5; i++) {
            canvas.drawCircle(100, startY + i * spacing, 40, paints[i]);
        }
        
        // Draw text
        canvas.drawText("Graphics Demo", 200, startY, paints[5]);
    }
}
```

## Drawing Basic Shapes

Android Canvas provides methods for drawing various geometric shapes.

### Lines and Points
```java
public class LinesAndPointsView extends View {
    private Paint linePaint, pointPaint;
    
    public LinesAndPointsView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setupPaints();
    }
    
    private void setupPaints() {
        linePaint = new Paint();
        linePaint.setColor(Color.BLUE);
        linePaint.setStrokeWidth(3f);
        linePaint.setAntiAlias(true);
        
        pointPaint = new Paint();
        pointPaint.setColor(Color.RED);
        pointPaint.setStrokeWidth(10f);
        pointPaint.setStrokeCap(Paint.Cap.ROUND);
        pointPaint.setAntiAlias(true);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        // Draw single line
        canvas.drawLine(50, 50, 300, 50, linePaint);
        
        // Draw multiple lines
        float[] lines = {
            50, 100, 300, 100,    // Line 1: start and end points
            50, 150, 300, 200,    // Line 2: start and end points
            100, 250, 250, 250   // Line 3: start and end points
        };
        canvas.drawLines(lines, linePaint);
        
        // Draw points
        float[] points = {100, 300, 150, 300, 200, 300, 250, 300};
        canvas.drawPoints(points, pointPaint);
        
        // Draw single point
        canvas.drawPoint(175, 350, pointPaint);
    }
}
```

### Rectangles and Rounded Rectangles
```java
public class RectanglesView extends View {
    private Paint fillPaint, strokePaint;
    private RectF rectF;
    
    public RectanglesView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setupPaints();
        rectF = new RectF();
    }
    
    private void setupPaints() {
        fillPaint = new Paint();
        fillPaint.setColor(Color.BLUE);
        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setAntiAlias(true);
        
        strokePaint = new Paint();
        strokePaint.setColor(Color.RED);
        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeWidth(4f);
        strokePaint.setAntiAlias(true);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        // Draw filled rectangle
        canvas.drawRect(50, 50, 200, 150, fillPaint);
        
        // Draw stroked rectangle
        canvas.drawRect(250, 50, 400, 150, strokePaint);
        
        // Draw rounded rectangle
        rectF.set(50, 200, 200, 300);
        canvas.drawRoundRect(rectF, 20f, 20f, fillPaint);
        
        // Draw oval in rectangle bounds
        rectF.set(250, 200, 400, 300);
        canvas.drawOval(rectF, strokePaint);
    }
}
```

### Circles and Arcs
```java
public class CirclesAndArcsView extends View {
    private Paint paint1, paint2, paint3;
    private RectF arcRect;
    
    public CirclesAndArcsView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setupPaints();
        arcRect = new RectF();
    }
    
    private void setupPaints() {
        paint1 = new Paint();
        paint1.setColor(Color.GREEN);
        paint1.setStyle(Paint.Style.FILL);
        paint1.setAntiAlias(true);
        
        paint2 = new Paint();
        paint2.setColor(Color.BLUE);
        paint2.setStyle(Paint.Style.STROKE);
        paint2.setStrokeWidth(6f);
        paint2.setAntiAlias(true);
        
        paint3 = new Paint();
        paint3.setColor(Color.RED);
        paint3.setStyle(Paint.Style.FILL);
        paint3.setAntiAlias(true);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        // Draw filled circle
        canvas.drawCircle(150, 150, 80, paint1);
        
        // Draw stroked circle
        canvas.drawCircle(350, 150, 80, paint2);
        
        // Draw arc (portion of circle)
        arcRect.set(100, 300, 200, 400);
        canvas.drawArc(arcRect, 0, 90, true, paint3); // 90-degree arc with center
        
        // Draw arc without center (curved line)
        arcRect.set(250, 300, 350, 400);
        canvas.drawArc(arcRect, 45, 180, false, paint2); // 180-degree arc without center
        
        // Draw complete oval
        arcRect.set(400, 300, 500, 400);
        canvas.drawOval(arcRect, paint1);
    }
}
```

## Path Drawing

Path allows creating complex shapes and curves by combining multiple drawing operations.

### Basic Path Operations
```java
public class BasicPathView extends View {
    private Paint pathPaint;
    private Path path;
    
    public BasicPathView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setupPaint();
        createPath();
    }
    
    private void setupPaint() {
        pathPaint = new Paint();
        pathPaint.setColor(Color.BLUE);
        pathPaint.setStyle(Paint.Style.STROKE);
        pathPaint.setStrokeWidth(4f);
        pathPaint.setStrokeJoin(Paint.Join.ROUND);
        pathPaint.setStrokeCap(Paint.Cap.ROUND);
        pathPaint.setAntiAlias(true);
    }
    
    private void createPath() {
        path = new Path();
        
        // Move to starting point
        path.moveTo(100, 100);
        
        // Draw lines to create a triangle
        path.lineTo(200, 100);
        path.lineTo(150, 50);
        path.close(); // Close the path back to starting point
        
        // Start a new subpath
        path.moveTo(250, 100);
        path.lineTo(350, 100);
        path.lineTo(350, 200);
        path.lineTo(250, 200);
        // Note: not closed, so it won't connect back to start
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        canvas.drawPath(path, pathPaint);
    }
}
```

### Advanced Path Operations
```java
public class AdvancedPathView extends View {
    private Paint fillPaint, strokePaint;
    private Path complexPath;
    
    public AdvancedPathView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setupPaints();
        createComplexPath();
    }
    
    private void setupPaints() {
        fillPaint = new Paint();
        fillPaint.setColor(Color.CYAN);
        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setAntiAlias(true);
        
        strokePaint = new Paint();
        strokePaint.setColor(Color.BLUE);
        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeWidth(3f);
        strokePaint.setAntiAlias(true);
    }
    
    private void createComplexPath() {
        complexPath = new Path();
        
        // Start with a move
        complexPath.moveTo(100, 200);
        
        // Add quadratic bezier curve
        complexPath.quadTo(150, 100, 200, 200);
        
        // Add cubic bezier curve
        complexPath.cubicTo(250, 150, 300, 250, 350, 200);
        
        // Add arc to path
        RectF arcRect = new RectF(350, 150, 450, 250);
        complexPath.arcTo(arcRect, 0, 90);
        
        // Add circle to path
        complexPath.addCircle(500, 200, 40, Path.Direction.CW);
        
        // Create a heart shape
        Path heartPath = new Path();
        heartPath.moveTo(100, 400);
        
        // Left curve of heart
        heartPath.cubicTo(100, 350, 50, 350, 50, 400);
        heartPath.cubicTo(50, 425, 75, 450, 100, 500);
        
        // Right curve of heart
        heartPath.cubicTo(125, 450, 150, 425, 150, 400);
        heartPath.cubicTo(150, 350, 100, 350, 100, 400);
        
        // Add heart to main path
        complexPath.addPath(heartPath);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        // Draw filled path
        canvas.drawPath(complexPath, fillPaint);
        
        // Draw stroke over it
        canvas.drawPath(complexPath, strokePaint);
    }
}
```

### Interactive Path Drawing
```java
public class DrawingPathView extends View {
    private Paint pathPaint;
    private Path currentPath;
    private List<Path> completedPaths;
    private float lastX, lastY;
    
    public DrawingPathView(Context context, AttributeSet attrs) {
        super(context, attrs);
        setupPaint();
        currentPath = new Path();
        completedPaths = new ArrayList<>();
    }
    
    private void setupPaint() {
        pathPaint = new Paint();
        pathPaint.setColor(Color.BLACK);
        pathPaint.setStyle(Paint.Style.STROKE);
        pathPaint.setStrokeWidth(5f);
        pathPaint.setStrokeJoin(Paint.Join.ROUND);
        pathPaint.setStrokeCap(Paint.Cap.ROUND);
        pathPaint.setAntiAlias(true);
    }
    
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();
        
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                currentPath.moveTo(x, y);
                lastX = x;
                lastY = y;
                return true;
                
            case MotionEvent.ACTION_MOVE:
                // Use quadratic curves for smoother drawing
                float midX = (x + lastX) / 2;
                float midY = (y + lastY) / 2;
                currentPath.quadTo(lastX, lastY, midX, midY);
                lastX = x;
                lastY = y;
                invalidate();
                break;
                
            case MotionEvent.ACTION_UP:
                // Finish the current path
                currentPath.lineTo(x, y);
                completedPaths.add(new Path(currentPath));
                currentPath.reset();
                invalidate();
                break;
        }
        
        return true;
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        // Draw all completed paths
        for (Path path : completedPaths) {
            canvas.drawPath(path, pathPaint);
        }
        
        // Draw current path being drawn
        canvas.drawPath(currentPath, pathPaint);
    }
    
    public void clearDrawing() {
        completedPaths.clear();
        currentPath.reset();
        invalidate();
    }
}
```

## Custom Views with Graphics

Creating custom views that use graphics for unique user interface elements.

### Custom Button with Graphics
```java
public class CustomGraphicsButton extends View {
    private Paint backgroundPaint, textPaint, borderPaint;
    private String buttonText = "Custom Button";
    private boolean isPressed = false;
    private RectF buttonRect;
    
    public CustomGraphicsButton(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        backgroundPaint = new Paint();
        backgroundPaint.setAntiAlias(true);
        
        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(48f);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);
        textPaint.setAntiAlias(true);
        
        borderPaint = new Paint();
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(4f);
        borderPaint.setAntiAlias(true);
        
        buttonRect = new RectF();
        
        setClickable(true);
    }
    
    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        
        int padding = 20;
        buttonRect.set(padding, padding, w - padding, h - padding);
        
        // Create gradient for button background
        LinearGradient gradient = new LinearGradient(
            0, padding, 0, h - padding,
            Color.parseColor("#4CAF50"), Color.parseColor("#2E7D32"),
            Shader.TileMode.CLAMP
        );
        backgroundPaint.setShader(gradient);
        borderPaint.setColor(Color.parseColor("#1B5E20"));
    }
    
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                isPressed = true;
                invalidate();
                return true;
                
            case MotionEvent.ACTION_UP:
                isPressed = false;
                invalidate();
                performClick();
                return true;
                
            case MotionEvent.ACTION_CANCEL:
                isPressed = false;
                invalidate();
                return true;
        }
        
        return super.onTouchEvent(event);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        // Apply press effect
        if (isPressed) {
            canvas.scale(0.95f, 0.95f, getWidth() / 2f, getHeight() / 2f);
        }
        
        // Draw button background
        canvas.drawRoundRect(buttonRect, 20f, 20f, backgroundPaint);
        
        // Draw border
        canvas.drawRoundRect(buttonRect, 20f, 20f, borderPaint);
        
        // Draw text
        float textX = getWidth() / 2f;
        float textY = getHeight() / 2f - ((textPaint.descent() + textPaint.ascent()) / 2);
        canvas.drawText(buttonText, textX, textY, textPaint);
    }
    
    public void setText(String text) {
        this.buttonText = text;
        invalidate();
    }
}
```

### Custom Progress View
```java
public class CustomProgressView extends View {
    private Paint backgroundPaint, progressPaint, textPaint;
    private RectF progressRect;
    private int progress = 0;
    private int maxProgress = 100;
    
    public CustomProgressView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        backgroundPaint = new Paint();
        backgroundPaint.setColor(Color.LTGRAY);
        backgroundPaint.setAntiAlias(true);
        
        progressPaint = new Paint();
        progressPaint.setAntiAlias(true);
        
        textPaint = new Paint();
        textPaint.setColor(Color.BLACK);
        textPaint.setTextSize(36f);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setAntiAlias(true);
        
        progressRect = new RectF();
    }
    
    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        
        progressRect.set(20, h / 2f - 30, w - 20, h / 2f + 30);
        
        // Create gradient for progress
        LinearGradient gradient = new LinearGradient(
            20, 0, w - 20, 0,
            new int[]{Color.RED, Color.YELLOW, Color.GREEN},
            null, Shader.TileMode.CLAMP
        );
        progressPaint.setShader(gradient);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        // Draw background
        canvas.drawRoundRect(progressRect, 15f, 15f, backgroundPaint);
        
        // Calculate progress width
        float progressWidth = (progressRect.width() * progress) / maxProgress;
        
        // Draw progress
        if (progressWidth > 0) {
            RectF currentProgressRect = new RectF(
                progressRect.left,
                progressRect.top,
                progressRect.left + progressWidth,
                progressRect.bottom
            );
            canvas.drawRoundRect(currentProgressRect, 15f, 15f, progressPaint);
        }
        
        // Draw progress text
        String progressText = progress + "%";
        float textX = getWidth() / 2f;
        float textY = progressRect.centerY() - ((textPaint.descent() + textPaint.ascent()) / 2);
        canvas.drawText(progressText, textX, textY, textPaint);
    }
    
    public void setProgress(int progress) {
        this.progress = Math.max(0, Math.min(progress, maxProgress));
        invalidate();
    }
    
    public int getProgress() {
        return progress;
    }
}
```

## Bitmap Operations

Working with bitmap images for advanced graphics manipulation.

### Bitmap Creation and Manipulation
```java
public class BitmapOperationsView extends View {
    private Bitmap originalBitmap, processedBitmap;
    private Paint paint;
    private Matrix matrix;
    
    public BitmapOperationsView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        paint = new Paint();
        paint.setAntiAlias(true);
        matrix = new Matrix();
        
        // Create a sample bitmap
        createSampleBitmap();
        
        // Process the bitmap
        processBitmap();
    }
    
    private void createSampleBitmap() {
        // Create a bitmap programmatically
        originalBitmap = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888);
        Canvas bitmapCanvas = new Canvas(originalBitmap);
        
        // Draw on the bitmap
        Paint bitmapPaint = new Paint();
        bitmapPaint.setAntiAlias(true);
        
        // Background
        bitmapPaint.setColor(Color.BLUE);
        bitmapCanvas.drawRect(0, 0, 200, 200, bitmapPaint);
        
        // Circle
        bitmapPaint.setColor(Color.YELLOW);
        bitmapCanvas.drawCircle(100, 100, 80, bitmapPaint);
        
        // Text
        bitmapPaint.setColor(Color.BLACK);
        bitmapPaint.setTextSize(24f);
        bitmapPaint.setTextAlign(Paint.Align.CENTER);
        bitmapCanvas.drawText("Sample", 100, 110, bitmapPaint);
    }
    
    private void processBitmap() {
        if (originalBitmap == null) return;
        
        // Create a copy for processing
        processedBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        
        // Apply color filter
        Canvas canvas = new Canvas(processedBitmap);
        Paint filterPaint = new Paint();
        
        // Create a color matrix for sepia effect
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0.5f); // Reduce saturation
        
        ColorMatrix sepiaMatrix = new ColorMatrix();
        sepiaMatrix.set(new float[]{
            0.393f, 0.769f, 0.189f, 0, 0,
            0.349f, 0.686f, 0.168f, 0, 0,
            0.272f, 0.534f, 0.131f, 0, 0,
            0, 0, 0, 1, 0
        });
        
        colorMatrix.postConcat(sepiaMatrix);
        filterPaint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));
        
        canvas.drawBitmap(originalBitmap, 0, 0, filterPaint);
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        if (originalBitmap != null) {
            // Draw original bitmap
            canvas.drawBitmap(originalBitmap, 50, 50, paint);
            
            // Draw text label
            Paint textPaint = new Paint();
            textPaint.setColor(Color.BLACK);
            textPaint.setTextSize(24f);
            canvas.drawText("Original", 50, 40, textPaint);
        }
        
        if (processedBitmap != null) {
            // Draw processed bitmap
            canvas.drawBitmap(processedBitmap, 300, 50, paint);
            
            // Draw text label
            Paint textPaint = new Paint();
            textPaint.setColor(Color.BLACK);
            textPaint.setTextSize(24f);
            canvas.drawText("Processed", 300, 40, textPaint);
        }
        
        // Demonstrate bitmap scaling
        if (originalBitmap != null) {
            matrix.reset();
            matrix.setScale(0.5f, 0.5f);
            matrix.postTranslate(50, 300);
            canvas.drawBitmap(originalBitmap, matrix, paint);
            
            Paint textPaint = new Paint();
            textPaint.setColor(Color.BLACK);
            textPaint.setTextSize(24f);
            canvas.drawText("Scaled 50%", 50, 290, textPaint);
        }
    }
}
```

## Matrix Transformations

Matrix operations allow for complex geometric transformations of graphics.

### Basic Matrix Operations
```java
public class MatrixTransformView extends View {
    private Paint paint;
    private Matrix matrix;
    private Path originalPath;
    private float rotation = 0f;
    
    public MatrixTransformView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        paint = new Paint();
        paint.setColor(Color.BLUE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4f);
        paint.setAntiAlias(true);
        
        matrix = new Matrix();
        
        // Create a simple path to transform
        originalPath = new Path();
        originalPath.moveTo(0, 0);
        originalPath.lineTo(100, 0);
        originalPath.lineTo(100, 100);
        originalPath.lineTo(0, 100);
        originalPath.close();
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        // Draw original path
        canvas.save();
        canvas.translate(100, 100);
        paint.setColor(Color.GRAY);
        canvas.drawPath(originalPath, paint);
        canvas.restore();
        
        // Apply various transformations
        demonstrateTransformations(canvas);
        
        // Animate rotation
        rotation += 2f;
        if (rotation >= 360f) rotation = 0f;
        invalidate();
    }
    
    private void demonstrateTransformations(Canvas canvas) {
        // Translation
        canvas.save();
        matrix.reset();
        matrix.setTranslate(250, 100);
        canvas.setMatrix(matrix);
        paint.setColor(Color.RED);
        canvas.drawPath(originalPath, paint);
        canvas.restore();
        
        // Scaling
        canvas.save();
        matrix.reset();
        matrix.setScale(1.5f, 1.5f);
        matrix.postTranslate(400, 50);
        canvas.setMatrix(matrix);
        paint.setColor(Color.GREEN);
        canvas.drawPath(originalPath, paint);
        canvas.restore();
        
        // Rotation
        canvas.save();
        matrix.reset();
        matrix.setRotate(rotation, 50, 50); // Rotate around center of shape
        matrix.postTranslate(100, 250);
        canvas.setMatrix(matrix);
        paint.setColor(Color.BLUE);
        canvas.drawPath(originalPath, paint);
        canvas.restore();
        
        // Skew
        canvas.save();
        matrix.reset();
        matrix.setSkew(0.2f, 0.1f);
        matrix.postTranslate(250, 250);
        canvas.setMatrix(matrix);
        paint.setColor(Color.MAGENTA);
        canvas.drawPath(originalPath, paint);
        canvas.restore();
        
        // Combined transformations
        canvas.save();
        matrix.reset();
        matrix.setScale(0.8f, 0.8f);
        matrix.postRotate(45f, 40, 40);
        matrix.postTranslate(400, 250);
        canvas.setMatrix(matrix);
        paint.setColor(Color.CYAN);
        canvas.drawPath(originalPath, paint);
        canvas.restore();
    }
}
```

## Animation with Graphics

Creating smooth animations using graphics and custom drawing.

### Animated Graphics View
```java
public class AnimatedGraphicsView extends View {
    private Paint paint;
    private float animationProgress = 0f;
    private ValueAnimator animator;
    private List<AnimatedShape> shapes;
    
    private static class AnimatedShape {
        float x, y, radius, speed;
        int color;
        
        AnimatedShape(float x, float y, float radius, float speed, int color) {
            this.x = x;
            this.y = y;
            this.radius = radius;
            this.speed = speed;
            this.color = color;
        }
    }
    
    public AnimatedGraphicsView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        paint = new Paint();
        paint.setAntiAlias(true);
        
        shapes = new ArrayList<>();
        initializeShapes();
        setupAnimation();
    }
    
    private void initializeShapes() {
        Random random = new Random();
        int[] colors = {Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.MAGENTA};
        
        for (int i = 0; i < 10; i++) {
            float x = random.nextFloat() * 800 + 100;
            float y = random.nextFloat() * 600 + 100;
            float radius = random.nextFloat() * 30 + 10;
            float speed = random.nextFloat() * 2 + 0.5f;
            int color = colors[random.nextInt(colors.length)];
            
            shapes.add(new AnimatedShape(x, y, radius, speed, color));
        }
    }
    
    private void setupAnimation() {
        animator = ValueAnimator.ofFloat(0f, 1f);
        animator.setDuration(2000);
        animator.setRepeatCount(ValueAnimator.INFINITE);
        animator.setRepeatMode(ValueAnimator.REVERSE);
        
        animator.addUpdateListener(animation -> {
            animationProgress = (Float) animation.getAnimatedValue();
            updateShapes();
            invalidate();
        });
        
        animator.start();
    }
    
    private void updateShapes() {
        for (AnimatedShape shape : shapes) {
            // Animate radius
            shape.radius = (float) (20 + 15 * Math.sin(animationProgress * Math.PI * 2));
            
            // Animate position
            shape.x += shape.speed * Math.cos(animationProgress * Math.PI * 4);
            shape.y += shape.speed * Math.sin(animationProgress * Math.PI * 3);
            
            // Bounce off edges
            if (shape.x < 0 || shape.x > getWidth()) shape.speed *= -1;
            if (shape.y < 0 || shape.y > getHeight()) shape.speed *= -1;
        }
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.BLACK);
        
        // Draw animated shapes
        for (AnimatedShape shape : shapes) {
            paint.setColor(shape.color);
            
            // Add transparency based on animation progress
            int alpha = (int) (255 * (0.3f + 0.7f * animationProgress));
            paint.setAlpha(alpha);
            
            canvas.drawCircle(shape.x, shape.y, shape.radius, paint);
        }
        
        // Draw connecting lines between nearby shapes
        paint.setColor(Color.WHITE);
        paint.setAlpha(100);
        paint.setStrokeWidth(2f);
        
        for (int i = 0; i < shapes.size(); i++) {
            for (int j = i + 1; j < shapes.size(); j++) {
                AnimatedShape shape1 = shapes.get(i);
                AnimatedShape shape2 = shapes.get(j);
                
                float distance = (float) Math.sqrt(
                    Math.pow(shape1.x - shape2.x, 2) + Math.pow(shape1.y - shape2.y, 2)
                );
                
                if (distance < 150) {
                    canvas.drawLine(shape1.x, shape1.y, shape2.x, shape2.y, paint);
                }
            }
        }
    }
    
    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        if (animator != null) {
            animator.cancel();
        }
    }
}
```

## Performance Optimization

Optimizing graphics performance for smooth user experience.

### Performance Best Practices
```java
public class OptimizedGraphicsView extends View {
    private Paint paint;
    private Path reusablePath;
    private RectF reusableRect;
    private Matrix reusableMatrix;
    
    // Pre-allocate objects to avoid garbage collection
    private static final int SHAPE_COUNT = 100;
    private float[] shapePositions = new float[SHAPE_COUNT * 2];
    private int[] shapeColors = new int[SHAPE_COUNT];
    
    public OptimizedGraphicsView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        paint = new Paint();
        paint.setAntiAlias(true);
        
        // Pre-allocate reusable objects
        reusablePath = new Path();
        reusableRect = new RectF();
        reusableMatrix = new Matrix();
        
        // Initialize shape data
        Random random = new Random();
        for (int i = 0; i < SHAPE_COUNT; i++) {
            shapePositions[i * 2] = random.nextFloat() * 1000;
            shapePositions[i * 2 + 1] = random.nextFloat() * 1000;
            shapeColors[i] = Color.HSVToColor(new float[]{
                random.nextFloat() * 360, 0.7f, 0.9f
            });
        }
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        canvas.drawColor(Color.WHITE);
        
        // Use efficient drawing methods
        drawOptimizedShapes(canvas);
    }
    
    private void drawOptimizedShapes(Canvas canvas) {
        // Batch similar operations together
        paint.setStyle(Paint.Style.FILL);
        
        for (int i = 0; i < SHAPE_COUNT; i++) {
            paint.setColor(shapeColors[i]);
            
            float x = shapePositions[i * 2];
            float y = shapePositions[i * 2 + 1];
            
            // Reuse rectangle object
            reusableRect.set(x, y, x + 20, y + 20);
            canvas.drawOval(reusableRect, paint);
        }
    }
    
    // Avoid allocations in onDraw
    private void efficientDrawing(Canvas canvas) {
        // BAD: Creates new objects in onDraw
        // Paint badPaint = new Paint();
        // RectF badRect = new RectF(0, 0, 100, 100);
        
        // GOOD: Reuse pre-allocated objects
        paint.setColor(Color.BLUE);
        reusableRect.set(0, 0, 100, 100);
        canvas.drawRect(reusableRect, paint);
    }
}
```

## Hardware Acceleration

Understanding and optimizing for hardware-accelerated graphics.

### Hardware Acceleration Considerations
Hardware acceleration is enabled by default for Android 4.0+ applications, but certain operations may fall back to software rendering.

### Supported Operations
- Most Canvas drawing operations
- Paint effects (shadows, gradients)
- Transformations (translate, rotate, scale)
- Clipping operations

### Unsupported Operations (Fall back to software)
- Canvas.drawPicture()
- Canvas.drawVertices()
- Canvas.drawTextOnPath()
- Some Paint effects (blur, subpixel text)

```java
// Check if hardware acceleration is enabled
if (canvas.isHardwareAccelerated()) {
    // Use hardware-accelerated operations
    drawWithHardwareAcceleration(canvas);
} else {
    // Fall back to software-optimized drawing
    drawWithSoftwareRendering(canvas);
}

// Disable hardware acceleration for specific view if needed
// In XML:
// android:layerType="software"

// In code:
// view.setLayerType(View.LAYER_TYPE_SOFTWARE, null);
```

## Best Practices

### Drawing Performance
- **Minimize onDraw() allocations**: Pre-allocate objects in constructor
- **Use appropriate Paint settings**: Set anti-aliasing only when needed
- **Batch similar operations**: Group drawing calls by Paint configuration
- **Avoid complex operations**: Keep onDraw() simple and fast
- **Use hardware acceleration**: Ensure drawing operations are GPU-friendly

### Memory Management
- **Recycle bitmaps**: Call bitmap.recycle() when done
- **Use appropriate bitmap formats**: Choose RGB_565 for photos, ARGB_8888 for graphics with transparency
- **Scale bitmaps appropriately**: Don't load full-size images when thumbnails will do
- **Cache drawing objects**: Reuse Paint, Path, and other objects

### Code Organization
- **Separate drawing logic**: Keep drawing code separate from business logic
- **Use custom views appropriately**: Don't override onDraw() unnecessarily
- **Profile performance**: Use Android Studio GPU profiler
- **Test on various devices**: Ensure performance across different hardware

Understanding Android's graphics system enables creating rich, interactive user interfaces with smooth performance. Proper use of Canvas, Paint, and other graphics APIs allows for limitless creative possibilities in Android applications.

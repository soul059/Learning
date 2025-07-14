# AWT Graphics and Custom Drawing

## 1. Introduction to AWT Graphics

AWT provides powerful graphics capabilities through the Graphics class and Canvas component. You can create custom drawings, animations, and visual effects.

### Graphics Class Overview
The Graphics class provides methods for:
- **Drawing shapes**: lines, rectangles, ovals, polygons
- **Filling shapes**: with solid colors or patterns
- **Drawing text**: with different fonts and colors
- **Drawing images**: loading and displaying images
- **Clipping**: restricting drawing to specific areas

### Canvas Component
Canvas is a blank component that you can draw on by overriding the `paint()` method.

## 2. Basic Drawing Operations

### Drawing Shapes Example
```java
import java.awt.*;
import java.awt.event.*;

public class BasicDrawingDemo extends Frame {
    private DrawingCanvas canvas;
    
    public BasicDrawingDemo() {
        canvas = new DrawingCanvas();
        canvas.setBackground(Color.WHITE);
        
        add(canvas, BorderLayout.CENTER);
        configureFrame();
    }
    
    private class DrawingCanvas extends Canvas {
        @Override
        public void paint(Graphics g) {
            super.paint(g);
            
            // Set drawing properties
            g.setColor(Color.BLACK);
            g.setFont(new Font("Arial", Font.BOLD, 16));
            
            // Draw title
            g.drawString("Basic AWT Drawing Operations", 20, 30);
            
            // Draw lines
            g.setColor(Color.RED);
            g.drawLine(50, 50, 200, 50);   // Horizontal line
            g.drawLine(50, 50, 50, 150);   // Vertical line
            g.drawLine(50, 150, 200, 50);  // Diagonal line
            
            // Draw rectangles
            g.setColor(Color.BLUE);
            g.drawRect(250, 50, 100, 60);      // Outline rectangle
            g.fillRect(250, 130, 100, 60);     // Filled rectangle
            
            // Draw rounded rectangles
            g.setColor(Color.GREEN);
            g.drawRoundRect(400, 50, 100, 60, 15, 15);  // Outline rounded rect
            g.fillRoundRect(400, 130, 100, 60, 15, 15); // Filled rounded rect
            
            // Draw ovals and circles
            g.setColor(Color.ORANGE);
            g.drawOval(50, 220, 100, 60);      // Oval
            g.fillOval(200, 220, 60, 60);      // Circle
            
            // Draw arcs
            g.setColor(Color.MAGENTA);
            g.drawArc(300, 220, 80, 80, 0, 90);    // Quarter circle
            g.fillArc(400, 220, 80, 80, 0, 180);   // Half circle
            
            // Draw polygons
            g.setColor(Color.CYAN);
            int[] xPoints = {550, 600, 575, 525};
            int[] yPoints = {50, 100, 150, 100};
            g.drawPolygon(xPoints, yPoints, 4);     // Outline polygon
            
            // Filled polygon (triangle)
            int[] triangleX = {550, 600, 575};
            int[] triangleY = {180, 180, 230};
            g.fillPolygon(triangleX, triangleY, 3);
            
            // Draw 3D rectangles
            g.setColor(Color.GRAY);
            g.draw3DRect(50, 320, 100, 40, true);   // Raised
            g.draw3DRect(200, 320, 100, 40, false); // Lowered
            
            // Different font styles
            g.setColor(Color.BLACK);
            g.setFont(new Font("Arial", Font.PLAIN, 12));
            g.drawString("Plain text", 350, 340);
            
            g.setFont(new Font("Arial", Font.BOLD, 12));
            g.drawString("Bold text", 350, 360);
            
            g.setFont(new Font("Arial", Font.ITALIC, 12));
            g.drawString("Italic text", 450, 340);
            
            g.setFont(new Font("Courier New", Font.BOLD, 12));
            g.drawString("Monospace text", 450, 360);
        }
    }
    
    private void configureFrame() {
        setTitle("Basic AWT Drawing Demo");
        setSize(700, 450);
        setLocationRelativeTo(null);
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new BasicDrawingDemo();
    }
}
```

### Interactive Drawing Application
```java
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

public class InteractiveDrawingApp extends Frame implements MouseListener, MouseMotionListener, ActionListener {
    private DrawingCanvas canvas;
    private Button clearButton, colorButton, circleButton, rectButton, lineButton;
    private Label statusLabel, coordLabel;
    private Choice thicknessChoice;
    private Checkbox fillCheckbox;
    
    // Drawing state
    private Color currentColor = Color.BLACK;
    private String drawingTool = "line";
    private int startX, startY, endX, endY;
    private boolean isDrawing = false;
    private int thickness = 1;
    private boolean fillShape = false;
    
    // Store drawn shapes
    private List<DrawnShape> shapes = new ArrayList<>();
    
    public InteractiveDrawingApp() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        canvas = new DrawingCanvas();
        canvas.setBackground(Color.WHITE);
        canvas.addMouseListener(this);
        canvas.addMouseMotionListener(this);
        
        // Control buttons
        clearButton = new Button("Clear");
        colorButton = new Button("Color");
        circleButton = new Button("Circle");
        rectButton = new Button("Rectangle");
        lineButton = new Button("Line");
        
        clearButton.addActionListener(this);
        colorButton.addActionListener(this);
        circleButton.addActionListener(this);
        rectButton.addActionListener(this);
        lineButton.addActionListener(this);
        
        // Set initial tool
        lineButton.setBackground(Color.YELLOW);
        
        // Thickness choice
        thicknessChoice = new Choice();
        thicknessChoice.add("1 px");
        thicknessChoice.add("2 px");
        thicknessChoice.add("3 px");
        thicknessChoice.add("5 px");
        thicknessChoice.add("8 px");
        thicknessChoice.addItemListener(e -> {
            String selected = thicknessChoice.getSelectedItem();
            thickness = Integer.parseInt(selected.substring(0, selected.indexOf(' ')));
        });
        
        // Fill checkbox
        fillCheckbox = new Checkbox("Fill", false);
        fillCheckbox.addItemListener(e -> fillShape = fillCheckbox.getState());
        
        // Status labels
        statusLabel = new Label("Select a tool and start drawing");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        coordLabel = new Label("Mouse: (0, 0)");
        coordLabel.setBackground(Color.LIGHT_GRAY);
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Toolbar
        Panel toolbar = new Panel(new FlowLayout());
        toolbar.add(new Label("Tools:"));
        toolbar.add(lineButton);
        toolbar.add(circleButton);
        toolbar.add(rectButton);
        toolbar.add(new Label(" | "));
        toolbar.add(colorButton);
        toolbar.add(new Label("Thickness:"));
        toolbar.add(thicknessChoice);
        toolbar.add(fillCheckbox);
        toolbar.add(new Label(" | "));
        toolbar.add(clearButton);
        
        // Status panel
        Panel statusPanel = new Panel(new GridLayout(1, 2));
        statusPanel.add(statusLabel);
        statusPanel.add(coordLabel);
        
        add(toolbar, BorderLayout.NORTH);
        add(canvas, BorderLayout.CENTER);
        add(statusPanel, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        // Reset button colors
        lineButton.setBackground(null);
        circleButton.setBackground(null);
        rectButton.setBackground(null);
        
        if (e.getSource() == clearButton) {
            shapes.clear();
            canvas.repaint();
            statusLabel.setText("Canvas cleared");
        } else if (e.getSource() == colorButton) {
            // Simple color cycling (in real app, use ColorDialog)
            Color[] colors = {Color.BLACK, Color.RED, Color.BLUE, Color.GREEN, 
                             Color.ORANGE, Color.MAGENTA, Color.CYAN};
            for (int i = 0; i < colors.length; i++) {
                if (currentColor.equals(colors[i])) {
                    currentColor = colors[(i + 1) % colors.length];
                    break;
                }
            }
            colorButton.setBackground(currentColor);
            statusLabel.setText("Color changed to " + getColorName(currentColor));
        } else if (e.getSource() == lineButton) {
            drawingTool = "line";
            lineButton.setBackground(Color.YELLOW);
            statusLabel.setText("Line tool selected");
        } else if (e.getSource() == circleButton) {
            drawingTool = "circle";
            circleButton.setBackground(Color.YELLOW);
            statusLabel.setText("Circle tool selected");
        } else if (e.getSource() == rectButton) {
            drawingTool = "rectangle";
            rectButton.setBackground(Color.YELLOW);
            statusLabel.setText("Rectangle tool selected");
        }
    }
    
    @Override
    public void mousePressed(MouseEvent e) {
        startX = e.getX();
        startY = e.getY();
        isDrawing = true;
        statusLabel.setText("Drawing " + drawingTool + "...");
    }
    
    @Override
    public void mouseReleased(MouseEvent e) {
        if (isDrawing) {
            endX = e.getX();
            endY = e.getY();
            
            // Create and store the shape
            DrawnShape shape = new DrawnShape(drawingTool, startX, startY, endX, endY, 
                                            currentColor, thickness, fillShape);
            shapes.add(shape);
            
            isDrawing = false;
            canvas.repaint();
            statusLabel.setText(drawingTool + " drawn at (" + startX + "," + startY + 
                              ") to (" + endX + "," + endY + ")");
        }
    }
    
    @Override
    public void mouseDragged(MouseEvent e) {
        if (isDrawing) {
            endX = e.getX();
            endY = e.getY();
            canvas.repaint(); // Show preview
        }
        coordLabel.setText("Mouse: (" + e.getX() + ", " + e.getY() + ")");
    }
    
    @Override
    public void mouseMoved(MouseEvent e) {
        coordLabel.setText("Mouse: (" + e.getX() + ", " + e.getY() + ")");
    }
    
    @Override
    public void mouseClicked(MouseEvent e) {}
    
    @Override
    public void mouseEntered(MouseEvent e) {}
    
    @Override
    public void mouseExited(MouseEvent e) {}
    
    private String getColorName(Color color) {
        if (color.equals(Color.BLACK)) return "Black";
        if (color.equals(Color.RED)) return "Red";
        if (color.equals(Color.BLUE)) return "Blue";
        if (color.equals(Color.GREEN)) return "Green";
        if (color.equals(Color.ORANGE)) return "Orange";
        if (color.equals(Color.MAGENTA)) return "Magenta";
        if (color.equals(Color.CYAN)) return "Cyan";
        return "Unknown";
    }
    
    // Inner class for drawing canvas
    private class DrawingCanvas extends Canvas {
        @Override
        public void paint(Graphics g) {
            super.paint(g);
            
            // Draw all stored shapes
            for (DrawnShape shape : shapes) {
                shape.draw(g);
            }
            
            // Draw preview of current shape being drawn
            if (isDrawing) {
                g.setColor(currentColor);
                drawShape(g, drawingTool, startX, startY, endX, endY, thickness, false);
            }
        }
        
        private void drawShape(Graphics g, String tool, int x1, int y1, int x2, int y2, 
                              int thick, boolean fill) {
            // Simple thickness simulation (real implementation would use Graphics2D)
            for (int i = 0; i < thick; i++) {
                switch (tool) {
                    case "line":
                        g.drawLine(x1 + i, y1, x2 + i, y2);
                        break;
                    case "rectangle":
                        int rectX = Math.min(x1, x2);
                        int rectY = Math.min(y1, y2);
                        int rectW = Math.abs(x2 - x1);
                        int rectH = Math.abs(y2 - y1);
                        if (fill && i == 0) {
                            g.fillRect(rectX, rectY, rectW, rectH);
                        } else {
                            g.drawRect(rectX + i, rectY + i, rectW - 2*i, rectH - 2*i);
                        }
                        break;
                    case "circle":
                        int circleX = Math.min(x1, x2);
                        int circleY = Math.min(y1, y2);
                        int circleW = Math.abs(x2 - x1);
                        int circleH = Math.abs(y2 - y1);
                        if (fill && i == 0) {
                            g.fillOval(circleX, circleY, circleW, circleH);
                        } else {
                            g.drawOval(circleX + i, circleY + i, circleW - 2*i, circleH - 2*i);
                        }
                        break;
                }
            }
        }
    }
    
    // Class to store drawn shapes
    private class DrawnShape {
        String tool;
        int x1, y1, x2, y2;
        Color color;
        int thickness;
        boolean fill;
        
        DrawnShape(String tool, int x1, int y1, int x2, int y2, Color color, int thickness, boolean fill) {
            this.tool = tool;
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.color = color;
            this.thickness = thickness;
            this.fill = fill;
        }
        
        void draw(Graphics g) {
            g.setColor(color);
            
            // Draw with thickness
            for (int i = 0; i < thickness; i++) {
                switch (tool) {
                    case "line":
                        g.drawLine(x1 + i, y1, x2 + i, y2);
                        break;
                    case "rectangle":
                        int rectX = Math.min(x1, x2);
                        int rectY = Math.min(y1, y2);
                        int rectW = Math.abs(x2 - x1);
                        int rectH = Math.abs(y2 - y1);
                        if (fill && i == 0) {
                            g.fillRect(rectX, rectY, rectW, rectH);
                        } else {
                            g.drawRect(rectX + i, rectY + i, rectW - 2*i, rectH - 2*i);
                        }
                        break;
                    case "circle":
                        int circleX = Math.min(x1, x2);
                        int circleY = Math.min(y1, y2);
                        int circleW = Math.abs(x2 - x1);
                        int circleH = Math.abs(y2 - y1);
                        if (fill && i == 0) {
                            g.fillOval(circleX, circleY, circleW, circleH);
                        } else {
                            g.drawOval(circleX + i, circleY + i, circleW - 2*i, circleH - 2*i);
                        }
                        break;
                }
            }
        }
    }
    
    private void configureFrame() {
        setTitle("Interactive Drawing Application");
        setSize(800, 600);
        setLocationRelativeTo(null);
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new InteractiveDrawingApp();
    }
}
```

## 3. Animation and Timer-based Graphics

### Simple Animation Demo
```java
import java.awt.*;
import java.awt.event.*;
import java.util.Timer;
import java.util.TimerTask;

public class AnimationDemo extends Frame implements ActionListener {
    private AnimationCanvas canvas;
    private Button startButton, stopButton, resetButton;
    private Label statusLabel;
    private Timer timer;
    private boolean isAnimating = false;
    
    public AnimationDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        canvas = new AnimationCanvas();
        canvas.setBackground(Color.BLACK);
        
        startButton = new Button("Start Animation");
        stopButton = new Button("Stop Animation");
        resetButton = new Button("Reset");
        
        startButton.addActionListener(this);
        stopButton.addActionListener(this);
        resetButton.addActionListener(this);
        
        statusLabel = new Label("Click Start to begin animation");
        statusLabel.setBackground(Color.LIGHT_GRAY);
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        Panel controlPanel = new Panel(new FlowLayout());
        controlPanel.add(startButton);
        controlPanel.add(stopButton);
        controlPanel.add(resetButton);
        
        add(controlPanel, BorderLayout.NORTH);
        add(canvas, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == startButton) {
            startAnimation();
        } else if (e.getSource() == stopButton) {
            stopAnimation();
        } else if (e.getSource() == resetButton) {
            resetAnimation();
        }
    }
    
    private void startAnimation() {
        if (!isAnimating) {
            isAnimating = true;
            timer = new Timer();
            timer.scheduleAtFixedRate(new TimerTask() {
                @Override
                public void run() {
                    canvas.updateAnimation();
                    canvas.repaint();
                }
            }, 0, 50); // 20 FPS
            
            statusLabel.setText("Animation running...");
        }
    }
    
    private void stopAnimation() {
        if (isAnimating) {
            isAnimating = false;
            if (timer != null) {
                timer.cancel();
            }
            statusLabel.setText("Animation stopped");
        }
    }
    
    private void resetAnimation() {
        stopAnimation();
        canvas.reset();
        canvas.repaint();
        statusLabel.setText("Animation reset");
    }
    
    private class AnimationCanvas extends Canvas {
        private int ballX = 50, ballY = 50;
        private int ballVelX = 3, ballVelY = 2;
        private int ballSize = 30;
        private Color ballColor = Color.RED;
        
        private int frame = 0;
        private double angle = 0;
        
        @Override
        public void paint(Graphics g) {
            super.paint(g);
            
            int width = getWidth();
            int height = getHeight();
            
            // Clear background
            g.setColor(Color.BLACK);
            g.fillRect(0, 0, width, height);
            
            // Draw bouncing ball
            g.setColor(ballColor);
            g.fillOval(ballX - ballSize/2, ballY - ballSize/2, ballSize, ballSize);
            
            // Draw ball trail
            g.setColor(new Color(ballColor.getRed(), ballColor.getGreen(), ballColor.getBlue(), 50));
            for (int i = 1; i <= 5; i++) {
                int trailX = ballX - ballVelX * i * 2;
                int trailY = ballY - ballVelY * i * 2;
                int trailSize = ballSize - i * 3;
                if (trailSize > 0) {
                    g.fillOval(trailX - trailSize/2, trailY - trailSize/2, trailSize, trailSize);
                }
            }
            
            // Draw spinning shapes
            g.setColor(Color.YELLOW);
            for (int i = 0; i < 8; i++) {
                double spinAngle = angle + (i * Math.PI / 4);
                int centerX = width / 2;
                int centerY = height / 2;
                int radius = 100;
                
                int x = centerX + (int)(Math.cos(spinAngle) * radius);
                int y = centerY + (int)(Math.sin(spinAngle) * radius);
                
                g.fillOval(x - 8, y - 8, 16, 16);
            }
            
            // Draw sine wave
            g.setColor(Color.CYAN);
            for (int x = 0; x < width; x += 2) {
                int y = height - 100 + (int)(Math.sin((x + frame) * 0.02) * 30);
                g.fillOval(x, y, 3, 3);
            }
            
            // Draw frame counter
            g.setColor(Color.WHITE);
            g.setFont(new Font("Arial", Font.BOLD, 16));
            g.drawString("Frame: " + frame, 10, 25);
            g.drawString("Ball: (" + ballX + ", " + ballY + ")", 10, 45);
            g.drawString("Velocity: (" + ballVelX + ", " + ballVelY + ")", 10, 65);
        }
        
        public void updateAnimation() {
            frame++;
            angle += 0.1;
            
            // Update ball position
            ballX += ballVelX;
            ballY += ballVelY;
            
            // Bounce off walls
            if (ballX <= ballSize/2 || ballX >= getWidth() - ballSize/2) {
                ballVelX = -ballVelX;
                changeBallColor();
            }
            if (ballY <= ballSize/2 || ballY >= getHeight() - ballSize/2) {
                ballVelY = -ballVelY;
                changeBallColor();
            }
            
            // Keep ball in bounds
            ballX = Math.max(ballSize/2, Math.min(getWidth() - ballSize/2, ballX));
            ballY = Math.max(ballSize/2, Math.min(getHeight() - ballSize/2, ballY));
        }
        
        private void changeBallColor() {
            Color[] colors = {Color.RED, Color.GREEN, Color.BLUE, Color.ORANGE, 
                             Color.MAGENTA, Color.CYAN, Color.YELLOW};
            ballColor = colors[(int)(Math.random() * colors.length)];
        }
        
        public void reset() {
            ballX = 50;
            ballY = 50;
            ballVelX = 3;
            ballVelY = 2;
            ballColor = Color.RED;
            frame = 0;
            angle = 0;
        }
    }
    
    private void configureFrame() {
        setTitle("AWT Animation Demo");
        setSize(800, 600);
        setLocationRelativeTo(null);
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                stopAnimation();
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new AnimationDemo();
    }
}
```

## 4. Chart and Graph Drawing

### Simple Chart Application
```java
import java.awt.*;
import java.awt.event.*;

public class ChartDemo extends Frame implements ActionListener {
    private ChartCanvas canvas;
    private TextField dataField;
    private Button addButton, clearButton;
    private Choice chartTypeChoice;
    private Label statusLabel;
    private java.util.List<Integer> data = new java.util.ArrayList<>();
    
    public ChartDemo() {
        setupComponents();
        setupLayout();
        addSampleData();
        configureFrame();
    }
    
    private void setupComponents() {
        canvas = new ChartCanvas();
        canvas.setBackground(Color.WHITE);
        
        dataField = new TextField("25", 10);
        addButton = new Button("Add Data");
        clearButton = new Button("Clear Data");
        
        addButton.addActionListener(this);
        clearButton.addActionListener(this);
        
        chartTypeChoice = new Choice();
        chartTypeChoice.add("Bar Chart");
        chartTypeChoice.add("Line Chart");
        chartTypeChoice.add("Pie Chart");
        chartTypeChoice.addItemListener(e -> canvas.repaint());
        
        statusLabel = new Label("Enter values and select chart type");
        statusLabel.setBackground(Color.LIGHT_GRAY);
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        Panel controlPanel = new Panel(new FlowLayout());
        controlPanel.add(new Label("Value:"));
        controlPanel.add(dataField);
        controlPanel.add(addButton);
        controlPanel.add(new Label("Chart Type:"));
        controlPanel.add(chartTypeChoice);
        controlPanel.add(clearButton);
        
        add(controlPanel, BorderLayout.NORTH);
        add(canvas, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == addButton) {
            try {
                int value = Integer.parseInt(dataField.getText());
                data.add(value);
                dataField.setText("");
                canvas.repaint();
                statusLabel.setText("Added value: " + value + " (Total: " + data.size() + " values)");
            } catch (NumberFormatException ex) {
                statusLabel.setText("Please enter a valid number");
            }
        } else if (e.getSource() == clearButton) {
            data.clear();
            canvas.repaint();
            statusLabel.setText("Data cleared");
        }
    }
    
    private void addSampleData() {
        data.add(25);
        data.add(40);
        data.add(15);
        data.add(60);
        data.add(35);
        data.add(20);
    }
    
    private class ChartCanvas extends Canvas {
        @Override
        public void paint(Graphics g) {
            super.paint(g);
            
            if (data.isEmpty()) {
                g.setColor(Color.GRAY);
                g.setFont(new Font("Arial", Font.ITALIC, 16));
                g.drawString("No data to display", getWidth()/2 - 70, getHeight()/2);
                return;
            }
            
            String chartType = chartTypeChoice.getSelectedItem();
            
            switch (chartType) {
                case "Bar Chart":
                    drawBarChart(g);
                    break;
                case "Line Chart":
                    drawLineChart(g);
                    break;
                case "Pie Chart":
                    drawPieChart(g);
                    break;
            }
        }
        
        private void drawBarChart(Graphics g) {
            int width = getWidth();
            int height = getHeight();
            int margin = 50;
            int chartWidth = width - 2 * margin;
            int chartHeight = height - 2 * margin;
            
            // Draw title
            g.setColor(Color.BLACK);
            g.setFont(new Font("Arial", Font.BOLD, 16));
            g.drawString("Bar Chart", width/2 - 40, 25);
            
            // Draw axes
            g.drawLine(margin, height - margin, width - margin, height - margin); // X-axis
            g.drawLine(margin, margin, margin, height - margin); // Y-axis
            
            // Find max value for scaling
            int maxValue = data.stream().mapToInt(Integer::intValue).max().orElse(1);
            
            // Draw bars
            int barWidth = chartWidth / data.size();
            Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.ORANGE, 
                             Color.MAGENTA, Color.CYAN, Color.YELLOW, Color.PINK};
            
            for (int i = 0; i < data.size(); i++) {
                int value = data.get(i);
                int barHeight = (value * chartHeight) / maxValue;
                int x = margin + i * barWidth;
                int y = height - margin - barHeight;
                
                g.setColor(colors[i % colors.length]);
                g.fillRect(x + 2, y, barWidth - 4, barHeight);
                
                // Draw value on top of bar
                g.setColor(Color.BLACK);
                g.setFont(new Font("Arial", Font.PLAIN, 10));
                g.drawString(String.valueOf(value), x + barWidth/2 - 5, y - 5);
                
                // Draw index below bar
                g.drawString(String.valueOf(i + 1), x + barWidth/2 - 3, height - margin + 15);
            }
            
            // Draw scale on Y-axis
            g.setColor(Color.GRAY);
            for (int i = 0; i <= 5; i++) {
                int value = (maxValue * i) / 5;
                int y = height - margin - (chartHeight * i) / 5;
                g.drawString(String.valueOf(value), 10, y + 3);
                g.drawLine(margin - 5, y, margin, y);
            }
        }
        
        private void drawLineChart(Graphics g) {
            int width = getWidth();
            int height = getHeight();
            int margin = 50;
            int chartWidth = width - 2 * margin;
            int chartHeight = height - 2 * margin;
            
            // Draw title
            g.setColor(Color.BLACK);
            g.setFont(new Font("Arial", Font.BOLD, 16));
            g.drawString("Line Chart", width/2 - 40, 25);
            
            // Draw axes
            g.drawLine(margin, height - margin, width - margin, height - margin); // X-axis
            g.drawLine(margin, margin, margin, height - margin); // Y-axis
            
            // Find max value for scaling
            int maxValue = data.stream().mapToInt(Integer::intValue).max().orElse(1);
            
            // Draw grid
            g.setColor(Color.LIGHT_GRAY);
            for (int i = 1; i < data.size(); i++) {
                int x = margin + (i * chartWidth) / (data.size() - 1);
                g.drawLine(x, margin, x, height - margin);
            }
            
            // Draw line and points
            g.setColor(Color.BLUE);
            for (int i = 0; i < data.size() - 1; i++) {
                int x1 = margin + (i * chartWidth) / (data.size() - 1);
                int y1 = height - margin - (data.get(i) * chartHeight) / maxValue;
                int x2 = margin + ((i + 1) * chartWidth) / (data.size() - 1);
                int y2 = height - margin - (data.get(i + 1) * chartHeight) / maxValue;
                
                // Draw line segment
                g.drawLine(x1, y1, x2, y2);
            }
            
            // Draw data points
            for (int i = 0; i < data.size(); i++) {
                int x = margin + (i * chartWidth) / (data.size() - 1);
                int y = height - margin - (data.get(i) * chartHeight) / maxValue;
                
                g.setColor(Color.RED);
                g.fillOval(x - 4, y - 4, 8, 8);
                
                // Draw value
                g.setColor(Color.BLACK);
                g.setFont(new Font("Arial", Font.PLAIN, 10));
                g.drawString(String.valueOf(data.get(i)), x - 8, y - 8);
            }
        }
        
        private void drawPieChart(Graphics g) {
            int width = getWidth();
            int height = getHeight();
            int centerX = width / 2;
            int centerY = height / 2;
            int radius = Math.min(width, height) / 3;
            
            // Draw title
            g.setColor(Color.BLACK);
            g.setFont(new Font("Arial", Font.BOLD, 16));
            g.drawString("Pie Chart", width/2 - 35, 25);
            
            // Calculate total
            int total = data.stream().mapToInt(Integer::intValue).sum();
            
            // Draw pie slices
            Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.ORANGE, 
                             Color.MAGENTA, Color.CYAN, Color.YELLOW, Color.PINK};
            
            int currentAngle = 0;
            for (int i = 0; i < data.size(); i++) {
                int value = data.get(i);
                int arcAngle = (value * 360) / total;
                
                g.setColor(colors[i % colors.length]);
                g.fillArc(centerX - radius, centerY - radius, 
                         radius * 2, radius * 2, currentAngle, arcAngle);
                
                // Draw slice outline
                g.setColor(Color.BLACK);
                g.drawArc(centerX - radius, centerY - radius, 
                         radius * 2, radius * 2, currentAngle, arcAngle);
                
                // Draw label
                double labelAngle = Math.toRadians(currentAngle + arcAngle / 2);
                int labelX = centerX + (int)((radius + 20) * Math.cos(labelAngle));
                int labelY = centerY + (int)((radius + 20) * Math.sin(labelAngle));
                
                g.setFont(new Font("Arial", Font.PLAIN, 10));
                String label = value + " (" + (value * 100 / total) + "%)";
                g.drawString(label, labelX - 15, labelY);
                
                currentAngle += arcAngle;
            }
        }
    }
    
    private void configureFrame() {
        setTitle("AWT Chart Demo");
        setSize(700, 500);
        setLocationRelativeTo(null);
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new ChartDemo();
    }
}
```

## Summary

AWT Graphics and Custom Drawing provides powerful capabilities for creating visual applications:

### Key Graphics Concepts

1. **Graphics Class**: The main drawing interface
   - Drawing methods: lines, shapes, text, images
   - Color and font management
   - Coordinate system and transformations

2. **Canvas Component**: Custom drawing surface
   - Override `paint()` method for custom drawing
   - Handle mouse and keyboard events for interaction
   - Double buffering for smooth animations

3. **Animation Techniques**: Timer-based updates
   - Use java.util.Timer for regular updates
   - Call `repaint()` to trigger redraw
   - Manage animation state and frame counting

### Best Practices

1. **Performance**: Minimize drawing operations in paint()
2. **Responsiveness**: Use separate threads for animations
3. **Memory Management**: Dispose of graphics resources properly
4. **User Experience**: Provide visual feedback for interactions
5. **Cross-platform**: Test on different operating systems

### Common Applications

- **Drawing Programs**: Paint-like applications with tools
- **Data Visualization**: Charts, graphs, and dashboards
- **Games**: Simple 2D games and animations
- **Custom UI Components**: Specialized visual elements
- **Scientific Visualization**: Plotting and analysis tools

AWT graphics provides the foundation for creating rich visual applications, though modern Java applications often use Swing's more advanced graphics capabilities or dedicated graphics libraries for complex visualizations.

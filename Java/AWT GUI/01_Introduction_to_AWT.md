# Introduction to AWT (Abstract Window Toolkit)

## 1. What is AWT?

AWT (Abstract Window Toolkit) is Java's original GUI (Graphical User Interface) toolkit. It was introduced with Java 1.0 and provides the foundation for creating desktop applications with graphical interfaces.

### Key Characteristics
- **Platform-dependent**: Uses native OS components (heavyweight components)
- **Foundation for Swing**: Swing is built on top of AWT
- **Event-driven**: Uses delegation event model
- **Cross-platform**: Write once, run anywhere with native look

### AWT vs Swing Comparison
| Feature | AWT | Swing |
|---------|-----|-------|
| Components | Heavyweight (native) | Lightweight (Java-drawn) |
| Look & Feel | Native OS appearance | Pluggable Look & Feel |
| Performance | Faster (native rendering) | Slower (Java rendering) |
| Customization | Limited | Extensive |
| Components | Basic set | Rich component set |

## 2. AWT Architecture

### Component Hierarchy
```
java.awt.Component
├── Button
├── Canvas
├── Checkbox
├── Choice
├── Label
├── List
├── Scrollbar
├── TextComponent
│   ├── TextArea
│   └── TextField
└── Container
    ├── Panel
    ├── ScrollPane
    └── Window
        ├── Dialog
        │   └── FileDialog
        └── Frame
```

### Basic AWT Application Structure
```java
import java.awt.*;
import java.awt.event.*;

public class BasicAWTApp extends Frame implements ActionListener {
    private Button button;
    private Label label;
    
    public BasicAWTApp() {
        setupComponents();
        setupLayout();
        setupEvents();
        configureFrame();
    }
    
    private void setupComponents() {
        button = new Button("Click Me!");
        label = new Label("Hello AWT!");
    }
    
    private void setupLayout() {
        setLayout(new FlowLayout());
        add(label);
        add(button);
    }
    
    private void setupEvents() {
        button.addActionListener(this);
        
        // Window closing event
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
    }
    
    private void configureFrame() {
        setTitle("Basic AWT Application");
        setSize(300, 150);
        setLocationRelativeTo(null);
        setVisible(true);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == button) {
            label.setText("Button was clicked!");
        }
    }
    
    public static void main(String[] args) {
        new BasicAWTApp();
    }
}
```

## 3. Core AWT Components

### Frame - Top-level Window
```java
import java.awt.*;
import java.awt.event.*;

public class FrameDemo extends Frame {
    public FrameDemo() {
        // Frame properties
        setTitle("AWT Frame Demo");
        setSize(400, 300);
        setLocation(100, 100);
        setResizable(true);
        
        // Background color
        setBackground(Color.LIGHT_GRAY);
        
        // Layout
        setLayout(new BorderLayout());
        
        // Add some components
        add(new Label("North", Label.CENTER), BorderLayout.NORTH);
        add(new Label("South", Label.CENTER), BorderLayout.SOUTH);
        add(new Label("Center Content", Label.CENTER), BorderLayout.CENTER);
        add(new Label("West", Label.CENTER), BorderLayout.WEST);
        add(new Label("East", Label.CENTER), BorderLayout.EAST);
        
        // Window closing
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                dispose(); // Close this window
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new FrameDemo();
    }
}
```

### Panel - Container Component
```java
import java.awt.*;
import java.awt.event.*;

public class PanelDemo extends Frame {
    public PanelDemo() {
        setTitle("AWT Panel Demo");
        setSize(500, 400);
        setLayout(new BorderLayout());
        
        // Create panels with different backgrounds
        Panel topPanel = new Panel();
        topPanel.setBackground(Color.RED);
        topPanel.setLayout(new FlowLayout());
        topPanel.add(new Label("Top Panel"));
        topPanel.add(new Button("Button 1"));
        topPanel.add(new Button("Button 2"));
        
        Panel centerPanel = new Panel();
        centerPanel.setBackground(Color.GREEN);
        centerPanel.setLayout(new GridLayout(2, 2));
        centerPanel.add(new Button("Grid 1"));
        centerPanel.add(new Button("Grid 2"));
        centerPanel.add(new Button("Grid 3"));
        centerPanel.add(new Button("Grid 4"));
        
        Panel bottomPanel = new Panel();
        bottomPanel.setBackground(Color.BLUE);
        bottomPanel.setLayout(new FlowLayout());
        bottomPanel.add(new Label("Bottom Panel"));
        bottomPanel.add(new Checkbox("Option 1"));
        bottomPanel.add(new Checkbox("Option 2"));
        
        // Add panels to frame
        add(topPanel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new PanelDemo();
    }
}
```

### Canvas - Custom Drawing Component
```java
import java.awt.*;
import java.awt.event.*;

public class CanvasDemo extends Frame {
    private DrawingCanvas canvas;
    
    public CanvasDemo() {
        setTitle("AWT Canvas Demo");
        setSize(600, 500);
        setLayout(new BorderLayout());
        
        // Create custom canvas
        canvas = new DrawingCanvas();
        canvas.setBackground(Color.WHITE);
        
        // Control panel
        Panel controlPanel = new Panel();
        controlPanel.setLayout(new FlowLayout());
        
        Button clearButton = new Button("Clear");
        Button circleButton = new Button("Draw Circle");
        Button rectangleButton = new Button("Draw Rectangle");
        
        clearButton.addActionListener(e -> canvas.clear());
        circleButton.addActionListener(e -> canvas.drawCircle());
        rectangleButton.addActionListener(e -> canvas.drawRectangle());
        
        controlPanel.add(clearButton);
        controlPanel.add(circleButton);
        controlPanel.add(rectangleButton);
        
        add(canvas, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.SOUTH);
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    // Custom Canvas class
    private class DrawingCanvas extends Canvas {
        private boolean drawCircle = false;
        private boolean drawRect = false;
        
        @Override
        public void paint(Graphics g) {
            super.paint(g);
            
            // Set drawing properties
            g.setColor(Color.BLUE);
            
            if (drawCircle) {
                // Draw multiple circles
                for (int i = 0; i < 5; i++) {
                    int x = 50 + i * 80;
                    int y = 50 + i * 30;
                    g.drawOval(x, y, 60, 60);
                    g.setColor(new Color(i * 50, 100, 255 - i * 40));
                }
            }
            
            if (drawRect) {
                // Draw multiple rectangles
                g.setColor(Color.RED);
                for (int i = 0; i < 4; i++) {
                    int x = 100 + i * 60;
                    int y = 200 + i * 20;
                    g.fillRect(x, y, 80, 40);
                    g.setColor(new Color(255 - i * 60, i * 80, 100));
                }
            }
            
            // Always draw some decorative elements
            g.setColor(Color.BLACK);
            g.drawString("AWT Canvas Drawing Demo", 20, 20);
            
            // Draw coordinate grid
            g.setColor(Color.LIGHT_GRAY);
            for (int i = 0; i < getWidth(); i += 50) {
                g.drawLine(i, 0, i, getHeight());
            }
            for (int i = 0; i < getHeight(); i += 50) {
                g.drawLine(0, i, getWidth(), i);
            }
        }
        
        public void clear() {
            drawCircle = false;
            drawRect = false;
            repaint();
        }
        
        public void drawCircle() {
            drawCircle = true;
            repaint();
        }
        
        public void drawRectangle() {
            drawRect = true;
            repaint();
        }
    }
    
    public static void main(String[] args) {
        new CanvasDemo();
    }
}
```

## 4. Event Handling in AWT

### Delegation Event Model
AWT uses the delegation event model where:
- **Event Sources** generate events
- **Event Listeners** handle events
- **Event Objects** carry information about events

### Common Event Types
```java
import java.awt.*;
import java.awt.event.*;

public class EventHandlingDemo extends Frame implements ActionListener, 
                                                        ItemListener, 
                                                        TextListener,
                                                        MouseListener {
    
    private Button button;
    private Checkbox checkbox;
    private TextField textField;
    private TextArea textArea;
    private Label statusLabel;
    
    public EventHandlingDemo() {
        setupComponents();
        setupLayout();
        setupEvents();
        configureFrame();
    }
    
    private void setupComponents() {
        button = new Button("Click Me");
        checkbox = new Checkbox("Enable Feature");
        textField = new TextField("Type here...", 20);
        textArea = new TextArea("Event log:\n", 10, 40);
        textArea.setEditable(false);
        statusLabel = new Label("Ready");
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        Panel topPanel = new Panel(new FlowLayout());
        topPanel.add(new Label("Button:"));
        topPanel.add(button);
        topPanel.add(new Label("Checkbox:"));
        topPanel.add(checkbox);
        
        Panel centerPanel = new Panel(new BorderLayout());
        centerPanel.add(new Label("Text Field:"), BorderLayout.NORTH);
        centerPanel.add(textField, BorderLayout.CENTER);
        
        add(topPanel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
        add(textArea, BorderLayout.SOUTH);
        add(statusLabel, BorderLayout.SOUTH);
    }
    
    private void setupEvents() {
        // Button events
        button.addActionListener(this);
        
        // Checkbox events
        checkbox.addItemListener(this);
        
        // Text field events
        textField.addTextListener(this);
        textField.addActionListener(this); // Enter key
        
        // Mouse events on the frame
        addMouseListener(this);
        
        // Window closing
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                logEvent("Window closing event received");
                System.exit(0);
            }
        });
        
        // Key events
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                logEvent("Key pressed: " + KeyEvent.getKeyText(e.getKeyCode()));
            }
        });
        
        setFocusable(true); // Enable key events on frame
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == button) {
            logEvent("Button clicked!");
            statusLabel.setText("Button was clicked");
        } else if (e.getSource() == textField) {
            logEvent("Enter pressed in text field: " + textField.getText());
            statusLabel.setText("Text field submitted");
        }
    }
    
    @Override
    public void itemStateChanged(ItemEvent e) {
        if (e.getSource() == checkbox) {
            boolean checked = checkbox.getState();
            logEvent("Checkbox " + (checked ? "checked" : "unchecked"));
            statusLabel.setText("Checkbox is " + (checked ? "ON" : "OFF"));
        }
    }
    
    @Override
    public void textValueChanged(TextEvent e) {
        if (e.getSource() == textField) {
            logEvent("Text changed: " + textField.getText());
            statusLabel.setText("Text field modified");
        }
    }
    
    @Override
    public void mouseClicked(MouseEvent e) {
        logEvent("Mouse clicked at (" + e.getX() + ", " + e.getY() + ")");
    }
    
    @Override
    public void mousePressed(MouseEvent e) {
        logEvent("Mouse pressed at (" + e.getX() + ", " + e.getY() + ")");
    }
    
    @Override
    public void mouseReleased(MouseEvent e) {
        logEvent("Mouse released at (" + e.getX() + ", " + e.getY() + ")");
    }
    
    @Override
    public void mouseEntered(MouseEvent e) {
        statusLabel.setText("Mouse entered the frame");
    }
    
    @Override
    public void mouseExited(MouseEvent e) {
        statusLabel.setText("Mouse exited the frame");
    }
    
    private void logEvent(String event) {
        textArea.append(event + "\n");
        // Auto-scroll to bottom
        textArea.setCaretPosition(textArea.getText().length());
    }
    
    private void configureFrame() {
        setTitle("AWT Event Handling Demo");
        setSize(500, 400);
        setLocationRelativeTo(null);
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new EventHandlingDemo();
    }
}
```

## 5. Complete Calculator Application

Here's a comprehensive calculator application demonstrating multiple AWT concepts:

```java
import java.awt.*;
import java.awt.event.*;

public class AWTCalculator extends Frame implements ActionListener {
    private TextField display;
    private Panel buttonPanel;
    private double num1, num2, result;
    private String operator;
    private boolean startNewNumber;
    
    public AWTCalculator() {
        setupDisplay();
        setupButtons();
        setupLayout();
        configureFrame();
        
        num1 = 0;
        num2 = 0;
        result = 0;
        operator = "";
        startNewNumber = true;
    }
    
    private void setupDisplay() {
        display = new TextField("0");
        display.setEditable(false);
        display.setFont(new Font("Arial", Font.BOLD, 20));
        display.setBackground(Color.WHITE);
        display.setForeground(Color.BLACK);
    }
    
    private void setupButtons() {
        buttonPanel = new Panel(new GridLayout(5, 4, 2, 2));
        
        String[] buttonLabels = {
            "C", "CE", "Back", "/",
            "7", "8", "9", "*",
            "4", "5", "6", "-",
            "1", "2", "3", "+",
            "0", ".", "=", ""
        };
        
        for (String label : buttonLabels) {
            if (!label.isEmpty()) {
                Button button = new Button(label);
                button.setFont(new Font("Arial", Font.BOLD, 16));
                button.addActionListener(this);
                
                // Color coding for different button types
                if (label.matches("[0-9]")) {
                    button.setBackground(Color.LIGHT_GRAY);
                } else if (label.matches("[+\\-*/=]")) {
                    button.setBackground(Color.ORANGE);
                    button.setForeground(Color.WHITE);
                } else {
                    button.setBackground(Color.GRAY);
                    button.setForeground(Color.WHITE);
                }
                
                buttonPanel.add(button);
            } else {
                buttonPanel.add(new Label("")); // Empty space
            }
        }
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Display panel
        Panel displayPanel = new Panel(new BorderLayout());
        displayPanel.add(display, BorderLayout.CENTER);
        displayPanel.setBackground(Color.BLACK);
        
        // Main layout
        add(displayPanel, BorderLayout.NORTH);
        add(buttonPanel, BorderLayout.CENTER);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        String command = e.getActionCommand();
        
        try {
            if (command.matches("[0-9]")) {
                handleNumber(command);
            } else if (command.equals(".")) {
                handleDecimal();
            } else if (command.matches("[+\\-*/]")) {
                handleOperator(command);
            } else if (command.equals("=")) {
                handleEquals();
            } else if (command.equals("C")) {
                handleClear();
            } else if (command.equals("CE")) {
                handleClearEntry();
            } else if (command.equals("Back")) {
                handleBackspace();
            }
        } catch (Exception ex) {
            display.setText("Error");
            startNewNumber = true;
        }
    }
    
    private void handleNumber(String digit) {
        if (startNewNumber) {
            display.setText(digit);
            startNewNumber = false;
        } else {
            String currentText = display.getText();
            if (!currentText.equals("0")) {
                display.setText(currentText + digit);
            } else {
                display.setText(digit);
            }
        }
    }
    
    private void handleDecimal() {
        String currentText = display.getText();
        if (startNewNumber) {
            display.setText("0.");
            startNewNumber = false;
        } else if (!currentText.contains(".")) {
            display.setText(currentText + ".");
        }
    }
    
    private void handleOperator(String op) {
        if (!operator.isEmpty() && !startNewNumber) {
            handleEquals();
        }
        
        num1 = Double.parseDouble(display.getText());
        operator = op;
        startNewNumber = true;
    }
    
    private void handleEquals() {
        if (!operator.isEmpty()) {
            num2 = Double.parseDouble(display.getText());
            
            switch (operator) {
                case "+":
                    result = num1 + num2;
                    break;
                case "-":
                    result = num1 - num2;
                    break;
                case "*":
                    result = num1 * num2;
                    break;
                case "/":
                    if (num2 != 0) {
                        result = num1 / num2;
                    } else {
                        throw new ArithmeticException("Division by zero");
                    }
                    break;
            }
            
            // Format result to remove unnecessary decimal places
            if (result == (long) result) {
                display.setText(String.valueOf((long) result));
            } else {
                display.setText(String.valueOf(result));
            }
            
            operator = "";
            startNewNumber = true;
        }
    }
    
    private void handleClear() {
        display.setText("0");
        num1 = 0;
        num2 = 0;
        result = 0;
        operator = "";
        startNewNumber = true;
    }
    
    private void handleClearEntry() {
        display.setText("0");
        startNewNumber = true;
    }
    
    private void handleBackspace() {
        String currentText = display.getText();
        if (currentText.length() > 1) {
            display.setText(currentText.substring(0, currentText.length() - 1));
        } else {
            display.setText("0");
            startNewNumber = true;
        }
    }
    
    private void configureFrame() {
        setTitle("AWT Calculator");
        setSize(300, 400);
        setLocationRelativeTo(null);
        setResizable(false);
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new AWTCalculator();
    }
}
```

## Summary

AWT provides the foundation for Java GUI programming with these key concepts:

### Core Components
- **Frame**: Top-level window container
- **Panel**: Lightweight container for organizing components
- **Canvas**: Custom drawing surface
- **Buttons, Labels, TextFields**: Basic interactive components

### Event Handling
- **Delegation Event Model**: Clean separation between event sources and listeners
- **Multiple Event Types**: Action, Item, Text, Mouse, Key, Window events
- **Event Adapters**: Convenience classes for handling multiple methods

### Key Advantages
- **Native Look**: Uses operating system's native components
- **Performance**: Direct OS rendering for better speed
- **Foundation Knowledge**: Understanding AWT helps with Swing development
- **Lightweight**: Smaller memory footprint than Swing

### Best Practices
- Always implement window closing events
- Use appropriate layout managers
- Handle exceptions in event handlers
- Separate component setup, layout, and event handling
- Use meaningful variable names and organize code logically

AWT remains relevant for understanding Java GUI fundamentals and for applications requiring native OS integration.

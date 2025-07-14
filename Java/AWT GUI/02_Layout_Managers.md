# AWT Layout Managers

## 1. Overview of Layout Managers

Layout managers in AWT automatically arrange components within containers. They handle component positioning, sizing, and behavior during window resizing.

### Why Use Layout Managers?
- **Automatic positioning**: No need to specify absolute coordinates
- **Responsive design**: Components adjust when window is resized
- **Cross-platform consistency**: Same layout works on different operating systems
- **Maintainable code**: Easier to modify and update layouts

### AWT Layout Manager Types
1. **FlowLayout** - Components flow left to right, top to bottom
2. **BorderLayout** - Five regions: North, South, East, West, Center
3. **GridLayout** - Uniform grid of rows and columns
4. **CardLayout** - Stack of components, one visible at a time
5. **GridBagLayout** - Most flexible, complex grid-based layout
6. **No Layout** (null) - Absolute positioning with setBounds()

## 2. FlowLayout

FlowLayout arranges components in a left-to-right flow, wrapping to the next line when necessary.

### Basic FlowLayout Example
```java
import java.awt.*;
import java.awt.event.*;

public class FlowLayoutDemo extends Frame {
    public FlowLayoutDemo() {
        // Set FlowLayout with different alignments
        setLayout(new FlowLayout(FlowLayout.CENTER, 10, 5));
        // FlowLayout(alignment, horizontal_gap, vertical_gap)
        
        // Add various components
        add(new Button("Button 1"));
        add(new Button("Button 2"));
        add(new Button("Long Button 3"));
        add(new Button("4"));
        add(new Label("Label"));
        add(new TextField("Text Field", 15));
        add(new Checkbox("Checkbox"));
        
        configureFrame();
    }
    
    private void configureFrame() {
        setTitle("FlowLayout Demo");
        setSize(400, 200);
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
        new FlowLayoutDemo();
    }
}
```

### FlowLayout Alignment Demo
```java
import java.awt.*;
import java.awt.event.*;

public class FlowLayoutAlignmentDemo extends Frame implements ActionListener {
    private Panel centerPanel;
    private Button leftAlign, centerAlign, rightAlign;
    
    public FlowLayoutAlignmentDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        // Control buttons
        leftAlign = new Button("Left Align");
        centerAlign = new Button("Center Align");
        rightAlign = new Button("Right Align");
        
        leftAlign.addActionListener(this);
        centerAlign.addActionListener(this);
        rightAlign.addActionListener(this);
        
        // Demo panel with components
        centerPanel = new Panel();
        centerPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 5, 5));
        
        // Add sample components to demo panel
        centerPanel.add(new Button("Sample 1"));
        centerPanel.add(new Button("Sample 2"));
        centerPanel.add(new Button("Sample 3"));
        centerPanel.add(new Label("Demo Label"));
        centerPanel.add(new TextField("Demo Text", 10));
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        Panel controlPanel = new Panel(new FlowLayout());
        controlPanel.add(leftAlign);
        controlPanel.add(centerAlign);
        controlPanel.add(rightAlign);
        
        add(controlPanel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == leftAlign) {
            centerPanel.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 5));
        } else if (e.getSource() == centerAlign) {
            centerPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 5, 5));
        } else if (e.getSource() == rightAlign) {
            centerPanel.setLayout(new FlowLayout(FlowLayout.RIGHT, 5, 5));
        }
        
        centerPanel.revalidate(); // Refresh layout
    }
    
    private void configureFrame() {
        setTitle("FlowLayout Alignment Demo");
        setSize(500, 250);
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
        new FlowLayoutAlignmentDemo();
    }
}
```

## 3. BorderLayout

BorderLayout divides the container into five regions: North, South, East, West, and Center.

### Basic BorderLayout Example
```java
import java.awt.*;
import java.awt.event.*;

public class BorderLayoutDemo extends Frame {
    public BorderLayoutDemo() {
        setLayout(new BorderLayout(5, 5)); // horizontal and vertical gaps
        
        // Add components to different regions
        add(new Button("North"), BorderLayout.NORTH);
        add(new Button("South"), BorderLayout.SOUTH);
        add(new Button("East"), BorderLayout.EAST);
        add(new Button("West"), BorderLayout.WEST);
        add(new Button("Center"), BorderLayout.CENTER);
        
        configureFrame();
    }
    
    private void configureFrame() {
        setTitle("BorderLayout Demo");
        setSize(400, 300);
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
        new BorderLayoutDemo();
    }
}
```

### Complex BorderLayout Application
```java
import java.awt.*;
import java.awt.event.*;

public class TextEditorDemo extends Frame implements ActionListener {
    private TextArea textArea;
    private TextField statusField;
    private MenuBar menuBar;
    private Menu fileMenu, editMenu;
    private MenuItem newItem, openItem, saveItem, exitItem;
    private MenuItem copyItem, pasteItem, selectAllItem;
    
    public TextEditorDemo() {
        setupMenus();
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupMenus() {
        menuBar = new MenuBar();
        
        // File menu
        fileMenu = new Menu("File");
        newItem = new MenuItem("New");
        openItem = new MenuItem("Open");
        saveItem = new MenuItem("Save");
        exitItem = new MenuItem("Exit");
        
        newItem.addActionListener(this);
        openItem.addActionListener(this);
        saveItem.addActionListener(this);
        exitItem.addActionListener(this);
        
        fileMenu.add(newItem);
        fileMenu.add(openItem);
        fileMenu.add(saveItem);
        fileMenu.addSeparator();
        fileMenu.add(exitItem);
        
        // Edit menu
        editMenu = new Menu("Edit");
        copyItem = new MenuItem("Copy");
        pasteItem = new MenuItem("Paste");
        selectAllItem = new MenuItem("Select All");
        
        copyItem.addActionListener(this);
        pasteItem.addActionListener(this);
        selectAllItem.addActionListener(this);
        
        editMenu.add(copyItem);
        editMenu.add(pasteItem);
        editMenu.addSeparator();
        editMenu.add(selectAllItem);
        
        menuBar.add(fileMenu);
        menuBar.add(editMenu);
        
        setMenuBar(menuBar);
    }
    
    private void setupComponents() {
        textArea = new TextArea("Welcome to AWT Text Editor!\nStart typing...");
        textArea.setFont(new Font("Courier New", Font.PLAIN, 14));
        
        statusField = new TextField("Ready");
        statusField.setEditable(false);
        statusField.setBackground(Color.LIGHT_GRAY);
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Toolbar (North)
        Panel toolbar = new Panel(new FlowLayout(FlowLayout.LEFT));
        toolbar.setBackground(Color.GRAY);
        
        Button newButton = new Button("New");
        Button openButton = new Button("Open");
        Button saveButton = new Button("Save");
        
        newButton.addActionListener(this);
        openButton.addActionListener(this);
        saveButton.addActionListener(this);
        
        toolbar.add(newButton);
        toolbar.add(openButton);
        toolbar.add(saveButton);
        
        // Side panel (West)
        Panel sidePanel = new Panel();
        sidePanel.setLayout(new GridLayout(5, 1, 0, 5));
        sidePanel.setBackground(Color.LIGHT_GRAY);
        sidePanel.add(new Label("Tools:"));
        sidePanel.add(new Button("Bold"));
        sidePanel.add(new Button("Italic"));
        sidePanel.add(new Button("Font"));
        sidePanel.add(new Button("Color"));
        
        add(toolbar, BorderLayout.NORTH);
        add(sidePanel, BorderLayout.WEST);
        add(textArea, BorderLayout.CENTER);
        add(statusField, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        String command = e.getActionCommand();
        
        switch (command) {
            case "New":
                textArea.setText("");
                statusField.setText("New document created");
                break;
            case "Open":
                statusField.setText("Open file dialog would appear here");
                break;
            case "Save":
                statusField.setText("File saved successfully");
                break;
            case "Exit":
                System.exit(0);
                break;
            case "Copy":
                textArea.copy();
                statusField.setText("Text copied to clipboard");
                break;
            case "Paste":
                textArea.paste();
                statusField.setText("Text pasted from clipboard");
                break;
            case "Select All":
                textArea.selectAll();
                statusField.setText("All text selected");
                break;
            default:
                statusField.setText("Action: " + command);
                break;
        }
    }
    
    private void configureFrame() {
        setTitle("AWT Text Editor - BorderLayout Demo");
        setSize(600, 400);
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
        new TextEditorDemo();
    }
}
```

## 4. GridLayout

GridLayout arranges components in a rectangular grid with equal-sized cells.

### Basic GridLayout Example
```java
import java.awt.*;
import java.awt.event.*;

public class GridLayoutDemo extends Frame {
    public GridLayoutDemo() {
        setLayout(new GridLayout(3, 4, 5, 5)); // 3 rows, 4 columns, gaps
        
        // Add numbered buttons
        for (int i = 1; i <= 12; i++) {
            Button button = new Button("Button " + i);
            add(button);
        }
        
        configureFrame();
    }
    
    private void configureFrame() {
        setTitle("GridLayout Demo - 3x4 Grid");
        setSize(400, 300);
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
        new GridLayoutDemo();
    }
}
```

### Calculator with GridLayout
```java
import java.awt.*;
import java.awt.event.*;

public class SimpleCalculator extends Frame implements ActionListener {
    private TextField display;
    private Panel buttonPanel;
    private String currentInput = "";
    private String operator = "";
    private double firstNumber = 0;
    private boolean newNumber = true;
    
    public SimpleCalculator() {
        setupDisplay();
        setupButtons();
        setupLayout();
        configureFrame();
    }
    
    private void setupDisplay() {
        display = new TextField("0");
        display.setEditable(false);
        display.setFont(new Font("Arial", Font.BOLD, 18));
        display.setBackground(Color.WHITE);
    }
    
    private void setupButtons() {
        buttonPanel = new Panel(new GridLayout(4, 4, 2, 2));
        
        String[] buttons = {
            "7", "8", "9", "/",
            "4", "5", "6", "*",
            "1", "2", "3", "-",
            "0", "C", "=", "+"
        };
        
        for (String text : buttons) {
            Button button = new Button(text);
            button.setFont(new Font("Arial", Font.BOLD, 16));
            button.addActionListener(this);
            
            // Color code different button types
            if (text.matches("[0-9]")) {
                button.setBackground(Color.LIGHT_GRAY);
            } else if (text.matches("[+\\-*/=]")) {
                button.setBackground(Color.ORANGE);
            } else {
                button.setBackground(Color.RED);
                button.setForeground(Color.WHITE);
            }
            
            buttonPanel.add(button);
        }
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        add(display, BorderLayout.NORTH);
        add(buttonPanel, BorderLayout.CENTER);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        String command = e.getActionCommand();
        
        if (command.matches("[0-9]")) {
            if (newNumber) {
                display.setText(command);
                newNumber = false;
            } else {
                display.setText(display.getText() + command);
            }
        } else if (command.equals("C")) {
            display.setText("0");
            currentInput = "";
            operator = "";
            firstNumber = 0;
            newNumber = true;
        } else if (command.matches("[+\\-*/]")) {
            firstNumber = Double.parseDouble(display.getText());
            operator = command;
            newNumber = true;
        } else if (command.equals("=")) {
            if (!operator.isEmpty()) {
                double secondNumber = Double.parseDouble(display.getText());
                double result = 0;
                
                switch (operator) {
                    case "+":
                        result = firstNumber + secondNumber;
                        break;
                    case "-":
                        result = firstNumber - secondNumber;
                        break;
                    case "*":
                        result = firstNumber * secondNumber;
                        break;
                    case "/":
                        if (secondNumber != 0) {
                            result = firstNumber / secondNumber;
                        } else {
                            display.setText("Error");
                            return;
                        }
                        break;
                }
                
                display.setText(String.valueOf(result));
                operator = "";
                newNumber = true;
            }
        }
    }
    
    private void configureFrame() {
        setTitle("Grid Calculator");
        setSize(250, 300);
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
        new SimpleCalculator();
    }
}
```

## 5. CardLayout

CardLayout stacks components like cards, showing only one at a time.

### CardLayout Demo
```java
import java.awt.*;
import java.awt.event.*;

public class CardLayoutDemo extends Frame implements ActionListener {
    private CardLayout cardLayout;
    private Panel cardPanel;
    private Button firstButton, secondButton, thirdButton;
    private Button nextButton, prevButton;
    private int currentCard = 0;
    private String[] cardNames = {"First", "Second", "Third"};
    
    public CardLayoutDemo() {
        setupCards();
        setupControls();
        setupLayout();
        configureFrame();
    }
    
    private void setupCards() {
        cardLayout = new CardLayout();
        cardPanel = new Panel(cardLayout);
        
        // First card
        Panel firstCard = new Panel(new FlowLayout());
        firstCard.setBackground(Color.RED);
        firstCard.add(new Label("This is the FIRST card"));
        firstCard.add(new Button("Card 1 Button"));
        firstCard.add(new TextField("First card text", 15));
        
        // Second card
        Panel secondCard = new Panel(new GridLayout(3, 2));
        secondCard.setBackground(Color.GREEN);
        secondCard.add(new Label("Second Card"));
        secondCard.add(new Button("Button A"));
        secondCard.add(new Label("More content"));
        secondCard.add(new Button("Button B"));
        secondCard.add(new Checkbox("Option 1"));
        secondCard.add(new Checkbox("Option 2"));
        
        // Third card
        Panel thirdCard = new Panel(new BorderLayout());
        thirdCard.setBackground(Color.BLUE);
        thirdCard.add(new Label("Third Card - BorderLayout", Label.CENTER), BorderLayout.NORTH);
        thirdCard.add(new TextArea("This is a text area in the third card.\nYou can type here!"), BorderLayout.CENTER);
        thirdCard.add(new Label("South section", Label.CENTER), BorderLayout.SOUTH);
        
        cardPanel.add(firstCard, cardNames[0]);
        cardPanel.add(secondCard, cardNames[1]);
        cardPanel.add(thirdCard, cardNames[2]);
    }
    
    private void setupControls() {
        firstButton = new Button("Show First");
        secondButton = new Button("Show Second");
        thirdButton = new Button("Show Third");
        nextButton = new Button("Next >");
        prevButton = new Button("< Previous");
        
        firstButton.addActionListener(this);
        secondButton.addActionListener(this);
        thirdButton.addActionListener(this);
        nextButton.addActionListener(this);
        prevButton.addActionListener(this);
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Top control panel
        Panel topPanel = new Panel(new FlowLayout());
        topPanel.add(firstButton);
        topPanel.add(secondButton);
        topPanel.add(thirdButton);
        
        // Bottom navigation panel
        Panel bottomPanel = new Panel(new FlowLayout());
        bottomPanel.add(prevButton);
        bottomPanel.add(new Label("Navigate:"));
        bottomPanel.add(nextButton);
        
        add(topPanel, BorderLayout.NORTH);
        add(cardPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == firstButton) {
            cardLayout.show(cardPanel, cardNames[0]);
            currentCard = 0;
        } else if (e.getSource() == secondButton) {
            cardLayout.show(cardPanel, cardNames[1]);
            currentCard = 1;
        } else if (e.getSource() == thirdButton) {
            cardLayout.show(cardPanel, cardNames[2]);
            currentCard = 2;
        } else if (e.getSource() == nextButton) {
            currentCard = (currentCard + 1) % cardNames.length;
            cardLayout.show(cardPanel, cardNames[currentCard]);
        } else if (e.getSource() == prevButton) {
            currentCard = (currentCard - 1 + cardNames.length) % cardNames.length;
            cardLayout.show(cardPanel, cardNames[currentCard]);
        }
    }
    
    private void configureFrame() {
        setTitle("CardLayout Demo");
        setSize(500, 400);
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
        new CardLayoutDemo();
    }
}
```

## 6. Null Layout (Absolute Positioning)

Sometimes you need precise control over component positioning using absolute coordinates.

### Absolute Positioning Example
```java
import java.awt.*;
import java.awt.event.*;

public class AbsolutePositioningDemo extends Frame implements ActionListener {
    private Button moveButton;
    private Label statusLabel;
    private int labelX = 50, labelY = 50;
    
    public AbsolutePositioningDemo() {
        setupComponents();
        configureFrame();
    }
    
    private void setupComponents() {
        setLayout(null); // No layout manager
        
        // Manually positioned components
        Label titleLabel = new Label("Absolute Positioning Demo");
        titleLabel.setBounds(20, 30, 200, 25);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 14));
        add(titleLabel);
        
        moveButton = new Button("Move Label");
        moveButton.setBounds(50, 100, 100, 30);
        moveButton.addActionListener(this);
        add(moveButton);
        
        statusLabel = new Label("Click to move me!");
        statusLabel.setBounds(labelX, labelY, 150, 25);
        statusLabel.setBackground(Color.YELLOW);
        add(statusLabel);
        
        // Some fixed positioned elements
        TextField textField = new TextField("Fixed text field");
        textField.setBounds(200, 100, 150, 25);
        add(textField);
        
        Checkbox checkbox = new Checkbox("Fixed checkbox");
        checkbox.setBounds(200, 140, 120, 25);
        add(checkbox);
        
        // Create a border around an area
        Panel borderPanel = new Panel();
        borderPanel.setBounds(20, 180, 330, 80);
        borderPanel.setBackground(Color.LIGHT_GRAY);
        borderPanel.setLayout(new FlowLayout());
        borderPanel.add(new Label("Panel with FlowLayout"));
        borderPanel.add(new Button("Button 1"));
        borderPanel.add(new Button("Button 2"));
        add(borderPanel);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == moveButton) {
            // Move the label to a random position
            labelX = (int) (Math.random() * 250) + 20;
            labelY = (int) (Math.random() * 200) + 60;
            statusLabel.setBounds(labelX, labelY, 150, 25);
            statusLabel.setText("Moved to (" + labelX + ", " + labelY + ")");
        }
    }
    
    private void configureFrame() {
        setTitle("Absolute Positioning Demo");
        setSize(400, 300);
        setLocationRelativeTo(null);
        setResizable(false); // Fixed size for absolute positioning
        
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        
        setVisible(true);
    }
    
    public static void main(String[] args) {
        new AbsolutePositioningDemo();
    }
}
```

## Summary

AWT Layout Managers provide powerful tools for creating responsive and maintainable user interfaces:

### Layout Manager Characteristics

| Layout Manager | Best Use Case | Advantages | Disadvantages |
|---------------|---------------|------------|---------------|
| **FlowLayout** | Simple horizontal arrangements | Easy to use, natural flow | Limited control, wrapping issues |
| **BorderLayout** | Main application windows | Clear regions, expandable center | Only 5 positions, components can be hidden |
| **GridLayout** | Uniform grids (calculators) | Equal-sized components | Inflexible, all cells same size |
| **CardLayout** | Wizard interfaces, tabbed content | Space-efficient, easy navigation | One component visible at a time |
| **Null Layout** | Precise positioning needs | Complete control | Not responsive, platform-dependent |

### Best Practices

1. **Choose the Right Layout**: Match layout manager to your specific needs
2. **Combine Layouts**: Use nested panels with different layouts for complex designs
3. **Handle Resizing**: Test how your layout behaves when window is resized
4. **Use Gaps**: Add spacing between components for better visual appeal
5. **Avoid Null Layout**: Use layout managers when possible for better maintainability
6. **Plan Your Structure**: Sketch your layout before coding

### Common Patterns

- **BorderLayout + FlowLayout**: Main window with toolbar (North) and content (Center)
- **GridLayout for Uniform Elements**: Calculators, button grids, forms
- **CardLayout for Navigation**: Settings panels, step-by-step wizards
- **Nested Panels**: Complex layouts using multiple layout managers

Understanding these layout managers is essential for creating professional-looking AWT applications that work consistently across different platforms and screen sizes.

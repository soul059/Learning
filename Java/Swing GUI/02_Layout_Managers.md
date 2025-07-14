# Layout Managers in Swing

## 1. Introduction to Layout Managers

Layout managers in Swing are responsible for determining the size and position of components within containers. They provide automatic component arrangement and handle resizing behavior.

### Why Use Layout Managers?
- **Automatic positioning**: Components are positioned automatically
- **Dynamic resizing**: Components adjust when container is resized
- **Platform independence**: Consistent appearance across different systems
- **Maintainability**: Easier to modify and maintain layouts

## 2. FlowLayout

FlowLayout arranges components in a left-to-right flow, wrapping to the next line when necessary.

### Basic FlowLayout Example
```java
import javax.swing.*;
import java.awt.*;

public class FlowLayoutDemo extends JFrame {
    
    public FlowLayoutDemo() {
        setupFlowLayout();
        configureFrame();
    }
    
    private void setupFlowLayout() {
        // Default FlowLayout (CENTER alignment, 5px gaps)
        setLayout(new FlowLayout());
        
        // Add multiple buttons
        for (int i = 1; i <= 8; i++) {
            add(new JButton("Button " + i));
        }
    }
    
    private void configureFrame() {
        setTitle("FlowLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 200);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new FlowLayoutDemo().setVisible(true);
        });
    }
}
```

### Advanced FlowLayout Configuration
```java
import javax.swing.*;
import java.awt.*;

public class FlowLayoutAdvanced extends JFrame {
    
    public FlowLayoutAdvanced() {
        createFlowLayoutVariations();
        configureFrame();
    }
    
    private void createFlowLayoutVariations() {
        setLayout(new BorderLayout());
        
        // Left-aligned FlowLayout
        JPanel leftPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 5));
        leftPanel.setBorder(BorderFactory.createTitledBorder("LEFT Alignment"));
        for (int i = 1; i <= 4; i++) {
            leftPanel.add(new JButton("L" + i));
        }
        
        // Center-aligned FlowLayout (default)
        JPanel centerPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 10));
        centerPanel.setBorder(BorderFactory.createTitledBorder("CENTER Alignment"));
        for (int i = 1; i <= 4; i++) {
            centerPanel.add(new JButton("C" + i));
        }
        
        // Right-aligned FlowLayout
        JPanel rightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 15));
        rightPanel.setBorder(BorderFactory.createTitledBorder("RIGHT Alignment"));
        for (int i = 1; i <= 4; i++) {
            rightPanel.add(new JButton("R" + i));
        }
        
        add(leftPanel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
        add(rightPanel, BorderLayout.SOUTH);
    }
    
    private void configureFrame() {
        setTitle("Advanced FlowLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(500, 300);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new FlowLayoutAdvanced().setVisible(true);
        });
    }
}
```

## 3. BorderLayout

BorderLayout divides the container into five regions: NORTH, SOUTH, EAST, WEST, and CENTER.

### Basic BorderLayout Example
```java
import javax.swing.*;
import java.awt.*;

public class BorderLayoutDemo extends JFrame {
    
    public BorderLayoutDemo() {
        setupBorderLayout();
        configureFrame();
    }
    
    private void setupBorderLayout() {
        setLayout(new BorderLayout());
        
        // Add components to different regions
        add(new JButton("NORTH"), BorderLayout.NORTH);
        add(new JButton("SOUTH"), BorderLayout.SOUTH);
        add(new JButton("EAST"), BorderLayout.EAST);
        add(new JButton("WEST"), BorderLayout.WEST);
        add(new JButton("CENTER"), BorderLayout.CENTER);
    }
    
    private void configureFrame() {
        setTitle("BorderLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 300);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new BorderLayoutDemo().setVisible(true);
        });
    }
}
```

### Advanced BorderLayout with Gaps and Complex Components
```java
import javax.swing.*;
import java.awt.*;

public class BorderLayoutAdvanced extends JFrame {
    
    public BorderLayoutAdvanced() {
        setupAdvancedBorderLayout();
        configureFrame();
    }
    
    private void setupAdvancedBorderLayout() {
        // BorderLayout with horizontal and vertical gaps
        setLayout(new BorderLayout(10, 10));
        
        // Create toolbar (NORTH)
        JToolBar toolBar = new JToolBar();
        toolBar.add(new JButton("New"));
        toolBar.add(new JButton("Open"));
        toolBar.add(new JButton("Save"));
        toolBar.addSeparator();
        toolBar.add(new JButton("Cut"));
        toolBar.add(new JButton("Copy"));
        toolBar.add(new JButton("Paste"));
        
        // Create sidebar (WEST)
        JPanel sidebar = new JPanel();
        sidebar.setLayout(new BoxLayout(sidebar, BoxLayout.Y_AXIS));
        sidebar.setBorder(BorderFactory.createTitledBorder("Tools"));
        sidebar.add(new JButton("Tool 1"));
        sidebar.add(new JButton("Tool 2"));
        sidebar.add(new JButton("Tool 3"));
        sidebar.setPreferredSize(new Dimension(100, 0));
        
        // Create main content area (CENTER)
        JTextArea textArea = new JTextArea();
        textArea.setText("This is the main content area.\nIt takes up the remaining space.");
        textArea.setBorder(BorderFactory.createTitledBorder("Content"));
        
        // Create properties panel (EAST)
        JPanel propertiesPanel = new JPanel(new GridLayout(5, 2, 5, 5));
        propertiesPanel.setBorder(BorderFactory.createTitledBorder("Properties"));
        propertiesPanel.add(new JLabel("Name:"));
        propertiesPanel.add(new JTextField());
        propertiesPanel.add(new JLabel("Size:"));
        propertiesPanel.add(new JTextField());
        propertiesPanel.add(new JLabel("Color:"));
        propertiesPanel.add(new JComboBox<>(new String[]{"Red", "Green", "Blue"}));
        propertiesPanel.add(new JLabel("Visible:"));
        propertiesPanel.add(new JCheckBox());
        propertiesPanel.add(new JButton("Apply"));
        propertiesPanel.add(new JButton("Reset"));
        propertiesPanel.setPreferredSize(new Dimension(150, 0));
        
        // Create status bar (SOUTH)
        JPanel statusBar = new JPanel(new FlowLayout(FlowLayout.LEFT));
        statusBar.setBorder(BorderFactory.createLoweredBevelBorder());
        statusBar.add(new JLabel("Ready"));
        statusBar.add(Box.createHorizontalStrut(20));
        statusBar.add(new JLabel("Line: 1, Column: 1"));
        
        // Add all components
        add(toolBar, BorderLayout.NORTH);
        add(sidebar, BorderLayout.WEST);
        add(new JScrollPane(textArea), BorderLayout.CENTER);
        add(propertiesPanel, BorderLayout.EAST);
        add(statusBar, BorderLayout.SOUTH);
    }
    
    private void configureFrame() {
        setTitle("Advanced BorderLayout - IDE-like Layout");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new BorderLayoutAdvanced().setVisible(true);
        });
    }
}
```

## 4. GridLayout

GridLayout arranges components in a rectangular grid where all components have the same size.

### Basic GridLayout Example
```java
import javax.swing.*;
import java.awt.*;

public class GridLayoutDemo extends JFrame {
    
    public GridLayoutDemo() {
        setupGridLayout();
        configureFrame();
    }
    
    private void setupGridLayout() {
        // Create 3x3 grid with 5px gaps
        setLayout(new GridLayout(3, 3, 5, 5));
        
        // Add buttons to fill the grid
        for (int i = 1; i <= 9; i++) {
            add(new JButton("Button " + i));
        }
    }
    
    private void configureFrame() {
        setTitle("GridLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 400);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new GridLayoutDemo().setVisible(true);
        });
    }
}
```

### Calculator Example using GridLayout
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class CalculatorDemo extends JFrame {
    private JTextField display;
    private double firstNumber = 0;
    private String operator = "";
    private boolean startNewNumber = true;
    
    public CalculatorDemo() {
        setupCalculator();
        configureFrame();
    }
    
    private void setupCalculator() {
        setLayout(new BorderLayout());
        
        // Create display
        display = new JTextField("0");
        display.setHorizontalAlignment(JTextField.RIGHT);
        display.setFont(new Font("Arial", Font.BOLD, 20));
        display.setEditable(false);
        display.setPreferredSize(new Dimension(0, 50));
        
        // Create button panel with GridLayout
        JPanel buttonPanel = new JPanel(new GridLayout(4, 4, 5, 5));
        
        // Button labels in calculator order
        String[] buttonLabels = {
            "7", "8", "9", "/",
            "4", "5", "6", "*",
            "1", "2", "3", "-",
            "0", ".", "=", "+"
        };
        
        // Create and add buttons
        for (String label : buttonLabels) {
            JButton button = new JButton(label);
            button.setFont(new Font("Arial", Font.BOLD, 18));
            button.addActionListener(new ButtonClickListener());
            
            // Color coding for different button types
            if (label.matches("[0-9.]")) {
                button.setBackground(Color.WHITE);
            } else if (label.equals("=")) {
                button.setBackground(Color.ORANGE);
            } else {
                button.setBackground(Color.LIGHT_GRAY);
            }
            
            buttonPanel.add(button);
        }
        
        // Add clear button
        JButton clearButton = new JButton("Clear");
        clearButton.setFont(new Font("Arial", Font.BOLD, 14));
        clearButton.setBackground(Color.RED);
        clearButton.setForeground(Color.WHITE);
        clearButton.addActionListener(e -> clearCalculator());
        
        JPanel topPanel = new JPanel(new BorderLayout());
        topPanel.add(display, BorderLayout.CENTER);
        topPanel.add(clearButton, BorderLayout.EAST);
        
        add(topPanel, BorderLayout.NORTH);
        add(buttonPanel, BorderLayout.CENTER);
    }
    
    private class ButtonClickListener implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            String command = e.getActionCommand();
            
            if (command.matches("[0-9]")) {
                handleNumber(command);
            } else if (command.equals(".")) {
                handleDecimal();
            } else if (command.matches("[+\\-*/]")) {
                handleOperator(command);
            } else if (command.equals("=")) {
                handleEquals();
            }
        }
    }
    
    private void handleNumber(String number) {
        if (startNewNumber) {
            display.setText(number);
            startNewNumber = false;
        } else {
            display.setText(display.getText() + number);
        }
    }
    
    private void handleDecimal() {
        if (startNewNumber) {
            display.setText("0.");
            startNewNumber = false;
        } else if (!display.getText().contains(".")) {
            display.setText(display.getText() + ".");
        }
    }
    
    private void handleOperator(String op) {
        if (!operator.isEmpty() && !startNewNumber) {
            handleEquals();
        }
        
        firstNumber = Double.parseDouble(display.getText());
        operator = op;
        startNewNumber = true;
    }
    
    private void handleEquals() {
        if (!operator.isEmpty() && !startNewNumber) {
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
            startNewNumber = true;
        }
    }
    
    private void clearCalculator() {
        display.setText("0");
        firstNumber = 0;
        operator = "";
        startNewNumber = true;
    }
    
    private void configureFrame() {
        setTitle("Calculator - GridLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(300, 400);
        setLocationRelativeTo(null);
        setResizable(false);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new CalculatorDemo().setVisible(true);
        });
    }
}
```

## 5. GridBagLayout

GridBagLayout is the most flexible layout manager, allowing precise control over component positioning and sizing.

### Basic GridBagLayout Example
```java
import javax.swing.*;
import java.awt.*;

public class GridBagLayoutDemo extends JFrame {
    
    public GridBagLayoutDemo() {
        setupGridBagLayout();
        configureFrame();
    }
    
    private void setupGridBagLayout() {
        setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        
        // Set default padding
        gbc.insets = new Insets(5, 5, 5, 5);
        
        // First row - single component spanning multiple columns
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.gridwidth = 3;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        add(new JLabel("Title (spans 3 columns)", JLabel.CENTER), gbc);
        
        // Second row - three equal components
        gbc.gridy = 1;
        gbc.gridwidth = 1;
        gbc.gridx = 0;
        add(new JButton("Button 1"), gbc);
        
        gbc.gridx = 1;
        add(new JButton("Button 2"), gbc);
        
        gbc.gridx = 2;
        add(new JButton("Button 3"), gbc);
        
        // Third row - component with different height
        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.gridwidth = 2;
        gbc.gridheight = 2;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        add(new JScrollPane(new JTextArea("Large text area\nspans 2x2 grid")), gbc);
        
        // Side component
        gbc.gridx = 2;
        gbc.gridy = 2;
        gbc.gridwidth = 1;
        gbc.gridheight = 1;
        gbc.weightx = 0;
        gbc.weighty = 0;
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.NORTH;
        add(new JButton("Side"), gbc);
        
        // Bottom component
        gbc.gridy = 3;
        add(new JButton("Bottom"), gbc);
    }
    
    private void configureFrame() {
        setTitle("GridBagLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(500, 400);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new GridBagLayoutDemo().setVisible(true);
        });
    }
}
```

### Complex Form using GridBagLayout
```java
import javax.swing.*;
import java.awt.*;

public class FormLayoutDemo extends JFrame {
    
    public FormLayoutDemo() {
        setupFormLayout();
        configureFrame();
    }
    
    private void setupFormLayout() {
        setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        
        // Form title
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        gbc.insets = new Insets(10, 10, 20, 10);
        gbc.anchor = GridBagConstraints.CENTER;
        JLabel titleLabel = new JLabel("User Registration Form");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 18));
        add(titleLabel, gbc);
        
        // Reset for form fields
        gbc.gridwidth = 1;
        gbc.insets = new Insets(5, 10, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        
        // First Name
        gbc.gridx = 0;
        gbc.gridy = 1;
        add(new JLabel("First Name:"), gbc);
        
        gbc.gridx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.insets = new Insets(5, 5, 5, 10);
        add(new JTextField(20), gbc);
        
        // Last Name
        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0;
        gbc.insets = new Insets(5, 10, 5, 5);
        add(new JLabel("Last Name:"), gbc);
        
        gbc.gridx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.insets = new Insets(5, 5, 5, 10);
        add(new JTextField(20), gbc);
        
        // Email
        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0;
        gbc.insets = new Insets(5, 10, 5, 5);
        add(new JLabel("Email:"), gbc);
        
        gbc.gridx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.insets = new Insets(5, 5, 5, 10);
        add(new JTextField(20), gbc);
        
        // Phone
        gbc.gridx = 0;
        gbc.gridy = 4;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0;
        gbc.insets = new Insets(5, 10, 5, 5);
        add(new JLabel("Phone:"), gbc);
        
        gbc.gridx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.insets = new Insets(5, 5, 5, 10);
        add(new JTextField(20), gbc);
        
        // Gender
        gbc.gridx = 0;
        gbc.gridy = 5;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0;
        gbc.insets = new Insets(5, 10, 5, 5);
        add(new JLabel("Gender:"), gbc);
        
        gbc.gridx = 1;
        gbc.insets = new Insets(5, 5, 5, 10);
        JPanel genderPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        ButtonGroup genderGroup = new ButtonGroup();
        JRadioButton maleRadio = new JRadioButton("Male");
        JRadioButton femaleRadio = new JRadioButton("Female");
        genderGroup.add(maleRadio);
        genderGroup.add(femaleRadio);
        genderPanel.add(maleRadio);
        genderPanel.add(femaleRadio);
        add(genderPanel, gbc);
        
        // Country
        gbc.gridx = 0;
        gbc.gridy = 6;
        gbc.insets = new Insets(5, 10, 5, 5);
        add(new JLabel("Country:"), gbc);
        
        gbc.gridx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(5, 5, 5, 10);
        JComboBox<String> countryCombo = new JComboBox<>(
            new String[]{"Select Country", "USA", "Canada", "UK", "India", "Australia"}
        );
        add(countryCombo, gbc);
        
        // Comments
        gbc.gridx = 0;
        gbc.gridy = 7;
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.NORTHWEST;
        gbc.insets = new Insets(5, 10, 5, 5);
        add(new JLabel("Comments:"), gbc);
        
        gbc.gridx = 1;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        gbc.insets = new Insets(5, 5, 5, 10);
        JTextArea commentsArea = new JTextArea(4, 20);
        commentsArea.setLineWrap(true);
        commentsArea.setWrapStyleWord(true);
        add(new JScrollPane(commentsArea), gbc);
        
        // Buttons
        gbc.gridx = 0;
        gbc.gridy = 8;
        gbc.gridwidth = 2;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0;
        gbc.weighty = 0;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.insets = new Insets(20, 10, 10, 10);
        
        JPanel buttonPanel = new JPanel(new FlowLayout());
        buttonPanel.add(new JButton("Submit"));
        buttonPanel.add(new JButton("Reset"));
        buttonPanel.add(new JButton("Cancel"));
        add(buttonPanel, gbc);
    }
    
    private void configureFrame() {
        setTitle("Form Layout - GridBagLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 500);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new FormLayoutDemo().setVisible(true);
        });
    }
}
```

## 6. BoxLayout

BoxLayout arranges components either horizontally or vertically in a single row or column.

### BoxLayout Example
```java
import javax.swing.*;
import java.awt.*;

public class BoxLayoutDemo extends JFrame {
    
    public BoxLayoutDemo() {
        setupBoxLayout();
        configureFrame();
    }
    
    private void setupBoxLayout() {
        setLayout(new BorderLayout());
        
        // Vertical BoxLayout
        JPanel verticalPanel = new JPanel();
        verticalPanel.setLayout(new BoxLayout(verticalPanel, BoxLayout.Y_AXIS));
        verticalPanel.setBorder(BorderFactory.createTitledBorder("Vertical BoxLayout"));
        
        verticalPanel.add(new JButton("Button 1"));
        verticalPanel.add(Box.createVerticalStrut(10)); // Fixed space
        verticalPanel.add(new JButton("Button 2"));
        verticalPanel.add(Box.createVerticalGlue()); // Flexible space
        verticalPanel.add(new JButton("Button 3"));
        verticalPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Fixed area
        verticalPanel.add(new JButton("Button 4"));
        
        // Horizontal BoxLayout
        JPanel horizontalPanel = new JPanel();
        horizontalPanel.setLayout(new BoxLayout(horizontalPanel, BoxLayout.X_AXIS));
        horizontalPanel.setBorder(BorderFactory.createTitledBorder("Horizontal BoxLayout"));
        
        horizontalPanel.add(new JButton("Left"));
        horizontalPanel.add(Box.createHorizontalStrut(10));
        horizontalPanel.add(new JButton("Center-Left"));
        horizontalPanel.add(Box.createHorizontalGlue());
        horizontalPanel.add(new JButton("Center-Right"));
        horizontalPanel.add(Box.createHorizontalStrut(10));
        horizontalPanel.add(new JButton("Right"));
        
        add(verticalPanel, BorderLayout.WEST);
        add(horizontalPanel, BorderLayout.CENTER);
    }
    
    private void configureFrame() {
        setTitle("BoxLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(600, 400);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new BoxLayoutDemo().setVisible(true);
        });
    }
}
```

## 7. CardLayout

CardLayout manages multiple components that share the same display space, showing only one at a time.

### CardLayout Example
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class CardLayoutDemo extends JFrame {
    private CardLayout cardLayout;
    private JPanel cardPanel;
    private int currentCard = 0;
    private String[] cardNames = {"Card 1", "Card 2", "Card 3", "Card 4"};
    
    public CardLayoutDemo() {
        setupCardLayout();
        configureFrame();
    }
    
    private void setupCardLayout() {
        setLayout(new BorderLayout());
        
        // Create CardLayout and panel
        cardLayout = new CardLayout();
        cardPanel = new JPanel(cardLayout);
        
        // Create different cards
        for (int i = 0; i < cardNames.length; i++) {
            JPanel card = createCard(i + 1);
            cardPanel.add(card, cardNames[i]);
        }
        
        // Create control panel
        JPanel controlPanel = new JPanel(new FlowLayout());
        
        JButton previousButton = new JButton("Previous");
        JButton nextButton = new JButton("Next");
        JButton firstButton = new JButton("First");
        JButton lastButton = new JButton("Last");
        
        JComboBox<String> cardSelector = new JComboBox<>(cardNames);
        
        // Add action listeners
        previousButton.addActionListener(e -> {
            cardLayout.previous(cardPanel);
            currentCard = (currentCard - 1 + cardNames.length) % cardNames.length;
            cardSelector.setSelectedIndex(currentCard);
        });
        
        nextButton.addActionListener(e -> {
            cardLayout.next(cardPanel);
            currentCard = (currentCard + 1) % cardNames.length;
            cardSelector.setSelectedIndex(currentCard);
        });
        
        firstButton.addActionListener(e -> {
            cardLayout.first(cardPanel);
            currentCard = 0;
            cardSelector.setSelectedIndex(currentCard);
        });
        
        lastButton.addActionListener(e -> {
            cardLayout.last(cardPanel);
            currentCard = cardNames.length - 1;
            cardSelector.setSelectedIndex(currentCard);
        });
        
        cardSelector.addActionListener(e -> {
            String selectedCard = (String) cardSelector.getSelectedItem();
            cardLayout.show(cardPanel, selectedCard);
            currentCard = cardSelector.getSelectedIndex();
        });
        
        controlPanel.add(firstButton);
        controlPanel.add(previousButton);
        controlPanel.add(cardSelector);
        controlPanel.add(nextButton);
        controlPanel.add(lastButton);
        
        add(cardPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.SOUTH);
    }
    
    private JPanel createCard(int cardNumber) {
        JPanel card = new JPanel(new BorderLayout());
        card.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        // Different background colors for visual distinction
        Color[] colors = {Color.LIGHT_GRAY, Color.CYAN, Color.YELLOW, Color.PINK};
        card.setBackground(colors[cardNumber - 1]);
        
        // Card title
        JLabel titleLabel = new JLabel("This is Card " + cardNumber, JLabel.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 24));
        
        // Card content
        JPanel contentPanel = new JPanel(new GridLayout(3, 1, 10, 10));
        contentPanel.setOpaque(false);
        
        contentPanel.add(new JLabel("Content for card " + cardNumber, JLabel.CENTER));
        contentPanel.add(new JButton("Action Button " + cardNumber));
        contentPanel.add(new JTextField("Text field " + cardNumber));
        
        card.add(titleLabel, BorderLayout.NORTH);
        card.add(contentPanel, BorderLayout.CENTER);
        
        return card;
    }
    
    private void configureFrame() {
        setTitle("CardLayout Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(500, 400);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new CardLayoutDemo().setVisible(true);
        });
    }
}
```

## 8. Layout Manager Comparison

### Comparison Table

| Layout Manager | Best Use Case | Complexity | Flexibility |
|---------------|---------------|------------|-------------|
| FlowLayout | Simple button bars, toolbars | Low | Low |
| BorderLayout | Main application layout | Medium | Medium |
| GridLayout | Uniform grids, calculators | Low | Low |
| GridBagLayout | Complex forms, precise control | High | Very High |
| BoxLayout | Simple linear arrangements | Medium | Medium |
| CardLayout | Wizards, tabbed interfaces | Medium | Medium |

### Choosing the Right Layout Manager

```java
import javax.swing.*;
import java.awt.*;

public class LayoutManagerComparison extends JFrame {
    
    public LayoutManagerComparison() {
        setupComparisonDemo();
        configureFrame();
    }
    
    private void setupComparisonDemo() {
        setLayout(new GridLayout(2, 3, 10, 10));
        
        // FlowLayout example
        JPanel flowPanel = new JPanel(new FlowLayout());
        flowPanel.setBorder(BorderFactory.createTitledBorder("FlowLayout"));
        flowPanel.add(new JButton("A"));
        flowPanel.add(new JButton("B"));
        flowPanel.add(new JButton("C"));
        
        // BorderLayout example
        JPanel borderPanel = new JPanel(new BorderLayout(5, 5));
        borderPanel.setBorder(BorderFactory.createTitledBorder("BorderLayout"));
        borderPanel.add(new JButton("N"), BorderLayout.NORTH);
        borderPanel.add(new JButton("S"), BorderLayout.SOUTH);
        borderPanel.add(new JButton("E"), BorderLayout.EAST);
        borderPanel.add(new JButton("W"), BorderLayout.WEST);
        borderPanel.add(new JButton("C"), BorderLayout.CENTER);
        
        // GridLayout example
        JPanel gridPanel = new JPanel(new GridLayout(2, 2, 5, 5));
        gridPanel.setBorder(BorderFactory.createTitledBorder("GridLayout"));
        gridPanel.add(new JButton("1"));
        gridPanel.add(new JButton("2"));
        gridPanel.add(new JButton("3"));
        gridPanel.add(new JButton("4"));
        
        // BoxLayout example
        JPanel boxPanel = new JPanel();
        boxPanel.setLayout(new BoxLayout(boxPanel, BoxLayout.Y_AXIS));
        boxPanel.setBorder(BorderFactory.createTitledBorder("BoxLayout"));
        boxPanel.add(new JButton("Top"));
        boxPanel.add(Box.createVerticalGlue());
        boxPanel.add(new JButton("Bottom"));
        
        // GridBagLayout example
        JPanel gbPanel = new JPanel(new GridBagLayout());
        gbPanel.setBorder(BorderFactory.createTitledBorder("GridBagLayout"));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.gridx = 0; gbc.gridy = 0;
        gbPanel.add(new JButton("1"), gbc);
        gbc.gridx = 1; gbc.gridwidth = 2;
        gbPanel.add(new JButton("2-3"), gbc);
        gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 3;
        gbPanel.add(new JButton("4-5-6"), gbc);
        
        // CardLayout example
        JPanel cardPanel = new JPanel(new CardLayout());
        cardPanel.setBorder(BorderFactory.createTitledBorder("CardLayout"));
        cardPanel.add(new JLabel("Card A", JLabel.CENTER), "A");
        cardPanel.add(new JLabel("Card B", JLabel.CENTER), "B");
        
        add(flowPanel);
        add(borderPanel);
        add(gridPanel);
        add(boxPanel);
        add(gbPanel);
        add(cardPanel);
    }
    
    private void configureFrame() {
        setTitle("Layout Manager Comparison");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new LayoutManagerComparison().setVisible(true);
        });
    }
}
```

## Summary

Layout managers are essential for creating professional and maintainable Swing applications:

### Key Points
- **FlowLayout**: Simple linear arrangement with wrapping
- **BorderLayout**: Five-region layout for main application structure
- **GridLayout**: Uniform grid with equal-sized components
- **GridBagLayout**: Most flexible with precise positioning control
- **BoxLayout**: Single-row or single-column arrangement
- **CardLayout**: Multiple components sharing same space

### Best Practices
- Choose the appropriate layout manager for each container
- Combine multiple layout managers for complex interfaces
- Use nested panels to achieve desired layouts
- Consider using GridBagLayout for forms and complex arrangements
- Test layouts with different window sizes and content
- Use borders and spacing for visual clarity

### Tips for Effective Layout Design
- Start with a rough sketch of your desired layout
- Break complex layouts into smaller, manageable panels
- Use consistent spacing and alignment
- Consider the user's workflow when arranging components
- Test layouts on different screen sizes and resolutions

# AWT Components

## 1. Overview of AWT Components

AWT provides a set of basic components for building user interfaces. These components are heavyweight, meaning they use native operating system widgets.

### Component Hierarchy
```
Component (abstract)
├── Button
├── Canvas
├── Checkbox
├── Choice
├── Label
├── List
├── Scrollbar
├── TextComponent (abstract)
│   ├── TextArea
│   └── TextField
└── Container (abstract)
    ├── Panel
    ├── ScrollPane
    └── Window (abstract)
        ├── Dialog
        │   └── FileDialog
        └── Frame
```

### Common Component Properties
All AWT components inherit these properties from the Component class:
- **Size and Position**: `setBounds()`, `setSize()`, `setLocation()`
- **Visibility**: `setVisible()`, `isVisible()`
- **Colors**: `setBackground()`, `setForeground()`
- **Fonts**: `setFont()`
- **Enabled State**: `setEnabled()`, `isEnabled()`

## 2. Button Component

Button is a clickable component that triggers actions when pressed.

### Basic Button Example
```java
import java.awt.*;
import java.awt.event.*;

public class ButtonDemo extends Frame implements ActionListener {
    private Button button1, button2, button3;
    private Label statusLabel;
    private int clickCount = 0;
    
    public ButtonDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        button1 = new Button("Simple Button");
        button2 = new Button("Colored Button");
        button3 = new Button("Disabled Button");
        
        // Customize button2
        button2.setBackground(Color.BLUE);
        button2.setForeground(Color.WHITE);
        button2.setFont(new Font("Arial", Font.BOLD, 14));
        
        // Disable button3
        button3.setEnabled(false);
        
        statusLabel = new Label("Click a button to see action");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        // Add action listeners
        button1.addActionListener(this);
        button2.addActionListener(this);
        button3.addActionListener(this);
    }
    
    private void setupLayout() {
        setLayout(new FlowLayout());
        add(button1);
        add(button2);
        add(button3);
        add(statusLabel);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        clickCount++;
        
        if (e.getSource() == button1) {
            statusLabel.setText("Simple button clicked! Count: " + clickCount);
            
            // Enable button3 after first click
            if (clickCount == 1) {
                button3.setEnabled(true);
                button3.setLabel("Now Enabled!");
            }
        } else if (e.getSource() == button2) {
            statusLabel.setText("Colored button clicked! Count: " + clickCount);
            
            // Change colors on each click
            Color[] colors = {Color.BLUE, Color.RED, Color.GREEN, Color.ORANGE};
            button2.setBackground(colors[clickCount % colors.length]);
        } else if (e.getSource() == button3) {
            statusLabel.setText("Previously disabled button clicked! Count: " + clickCount);
        }
    }
    
    private void configureFrame() {
        setTitle("Button Demo");
        setSize(400, 150);
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
        new ButtonDemo();
    }
}
```

### Interactive Button Game
```java
import java.awt.*;
import java.awt.event.*;

public class ButtonGameDemo extends Frame implements ActionListener {
    private Button[] gameButtons;
    private Button resetButton;
    private Label scoreLabel, instructionLabel;
    private int score = 0;
    private int targetButton;
    
    public ButtonGameDemo() {
        setupComponents();
        setupLayout();
        startNewRound();
        configureFrame();
    }
    
    private void setupComponents() {
        gameButtons = new Button[9];
        for (int i = 0; i < 9; i++) {
            gameButtons[i] = new Button("" + (i + 1));
            gameButtons[i].setFont(new Font("Arial", Font.BOLD, 20));
            gameButtons[i].addActionListener(this);
        }
        
        resetButton = new Button("Reset Game");
        resetButton.setBackground(Color.RED);
        resetButton.setForeground(Color.WHITE);
        resetButton.addActionListener(this);
        
        scoreLabel = new Label("Score: 0");
        scoreLabel.setFont(new Font("Arial", Font.BOLD, 16));
        
        instructionLabel = new Label("Click the highlighted button!");
        instructionLabel.setFont(new Font("Arial", Font.PLAIN, 14));
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Top panel with instructions and score
        Panel topPanel = new Panel(new FlowLayout());
        topPanel.add(instructionLabel);
        topPanel.add(scoreLabel);
        
        // Center panel with game buttons (3x3 grid)
        Panel gamePanel = new Panel(new GridLayout(3, 3, 5, 5));
        for (Button button : gameButtons) {
            gamePanel.add(button);
        }
        
        // Bottom panel with reset button
        Panel bottomPanel = new Panel(new FlowLayout());
        bottomPanel.add(resetButton);
        
        add(topPanel, BorderLayout.NORTH);
        add(gamePanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }
    
    private void startNewRound() {
        // Reset all button colors
        for (Button button : gameButtons) {
            button.setBackground(Color.LIGHT_GRAY);
        }
        
        // Randomly select target button
        targetButton = (int) (Math.random() * 9);
        gameButtons[targetButton].setBackground(Color.YELLOW);
        
        instructionLabel.setText("Click button " + (targetButton + 1) + "!");
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == resetButton) {
            score = 0;
            scoreLabel.setText("Score: 0");
            startNewRound();
            return;
        }
        
        // Check if correct button was clicked
        for (int i = 0; i < gameButtons.length; i++) {
            if (e.getSource() == gameButtons[i]) {
                if (i == targetButton) {
                    score++;
                    scoreLabel.setText("Score: " + score);
                    instructionLabel.setText("Correct! Get ready for next round...");
                    
                    // Start new round after short delay
                    Timer timer = new Timer();
                    timer.schedule(new java.util.TimerTask() {
                        @Override
                        public void run() {
                            startNewRound();
                        }
                    }, 1000);
                } else {
                    instructionLabel.setText("Wrong button! Try again.");
                    gameButtons[i].setBackground(Color.RED);
                }
                break;
            }
        }
    }
    
    private void configureFrame() {
        setTitle("Button Click Game");
        setSize(350, 400);
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
        new ButtonGameDemo();
    }
}
```

## 3. Label Component

Label displays non-editable text or images.

### Label Demo
```java
import java.awt.*;
import java.awt.event.*;

public class LabelDemo extends Frame implements ActionListener {
    private Label titleLabel, infoLabel, dynamicLabel;
    private Button changeButton, alignButton;
    private int changeCount = 0;
    private int currentAlignment = Label.LEFT;
    
    public LabelDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        // Different types of labels
        titleLabel = new Label("Label Demo Application");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 18));
        titleLabel.setForeground(Color.BLUE);
        titleLabel.setAlignment(Label.CENTER);
        
        infoLabel = new Label("Labels can display text with different alignments and styles");
        infoLabel.setFont(new Font("Arial", Font.ITALIC, 12));
        
        dynamicLabel = new Label("This label changes dynamically");
        dynamicLabel.setBackground(Color.YELLOW);
        dynamicLabel.setAlignment(Label.LEFT);
        
        changeButton = new Button("Change Label Text");
        changeButton.addActionListener(this);
        
        alignButton = new Button("Change Alignment");
        alignButton.addActionListener(this);
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        Panel topPanel = new Panel(new FlowLayout());
        topPanel.add(titleLabel);
        
        Panel centerPanel = new Panel(new GridLayout(3, 1, 5, 10));
        centerPanel.add(infoLabel);
        centerPanel.add(dynamicLabel);
        
        Panel bottomPanel = new Panel(new FlowLayout());
        bottomPanel.add(changeButton);
        bottomPanel.add(alignButton);
        
        add(topPanel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == changeButton) {
            changeCount++;
            String[] messages = {
                "Text changed! Count: " + changeCount,
                "Dynamic content update #" + changeCount,
                "Label modification " + changeCount,
                "Text transformation " + changeCount
            };
            
            dynamicLabel.setText(messages[changeCount % messages.length]);
            
            // Change background color too
            Color[] colors = {Color.YELLOW, Color.PINK, Color.CYAN, Color.LIGHT_GRAY};
            dynamicLabel.setBackground(colors[changeCount % colors.length]);
            
        } else if (e.getSource() == alignButton) {
            currentAlignment++;
            if (currentAlignment > Label.RIGHT) {
                currentAlignment = Label.LEFT;
            }
            
            dynamicLabel.setAlignment(currentAlignment);
            
            String alignmentText = "";
            switch (currentAlignment) {
                case Label.LEFT:
                    alignmentText = "Left aligned";
                    break;
                case Label.CENTER:
                    alignmentText = "Center aligned";
                    break;
                case Label.RIGHT:
                    alignmentText = "Right aligned";
                    break;
            }
            
            infoLabel.setText("Current alignment: " + alignmentText);
        }
    }
    
    private void configureFrame() {
        setTitle("Label Component Demo");
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
        new LabelDemo();
    }
}
```

## 4. TextField Component

TextField provides single-line text input.

### TextField Demo
```java
import java.awt.*;
import java.awt.event.*;

public class TextFieldDemo extends Frame implements ActionListener, TextListener, FocusListener {
    private TextField nameField, ageField, emailField;
    private Label nameLabel, ageLabel, emailLabel, statusLabel;
    private Button submitButton, clearButton;
    private TextArea resultArea;
    
    public TextFieldDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        // Text fields with different properties
        nameField = new TextField(20);
        nameField.setFont(new Font("Arial", Font.PLAIN, 14));
        nameField.addActionListener(this); // Enter key
        nameField.addTextListener(this);   // Text changes
        nameField.addFocusListener(this);  // Focus events
        
        ageField = new TextField(5);
        ageField.addActionListener(this);
        ageField.addTextListener(this);
        ageField.addFocusListener(this);
        
        emailField = new TextField(25);
        emailField.addActionListener(this);
        emailField.addTextListener(this);
        emailField.addFocusListener(this);
        
        // Labels
        nameLabel = new Label("Name:");
        ageLabel = new Label("Age:");
        emailLabel = new Label("Email:");
        statusLabel = new Label("Enter your information above");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        // Buttons
        submitButton = new Button("Submit");
        submitButton.addActionListener(this);
        
        clearButton = new Button("Clear All");
        clearButton.addActionListener(this);
        
        // Result area
        resultArea = new TextArea(5, 40);
        resultArea.setEditable(false);
        resultArea.setFont(new Font("Courier New", Font.PLAIN, 12));
        resultArea.setText("Form data will appear here when submitted...\n");
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Form panel
        Panel formPanel = new Panel(new GridLayout(4, 2, 5, 5));
        formPanel.add(nameLabel);
        formPanel.add(nameField);
        formPanel.add(ageLabel);
        formPanel.add(ageField);
        formPanel.add(emailLabel);
        formPanel.add(emailField);
        
        // Button panel
        Panel buttonPanel = new Panel(new FlowLayout());
        buttonPanel.add(submitButton);
        buttonPanel.add(clearButton);
        
        formPanel.add(new Label("")); // Empty cell
        formPanel.add(buttonPanel);
        
        // Main layout
        add(formPanel, BorderLayout.NORTH);
        add(statusLabel, BorderLayout.CENTER);
        add(resultArea, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == submitButton || 
            e.getSource() == nameField || 
            e.getSource() == ageField || 
            e.getSource() == emailField) {
            
            submitForm();
            
        } else if (e.getSource() == clearButton) {
            clearForm();
        }
    }
    
    @Override
    public void textValueChanged(TextEvent e) {
        // Update status as user types
        TextField source = (TextField) e.getSource();
        String fieldName = "";
        
        if (source == nameField) fieldName = "Name";
        else if (source == ageField) fieldName = "Age";
        else if (source == emailField) fieldName = "Email";
        
        statusLabel.setText(fieldName + " field modified: " + source.getText());
        
        // Validate age field in real-time
        if (source == ageField) {
            validateAge();
        }
        
        // Validate email field
        if (source == emailField) {
            validateEmail();
        }
    }
    
    @Override
    public void focusGained(FocusEvent e) {
        TextField source = (TextField) e.getSource();
        source.setBackground(Color.YELLOW); // Highlight focused field
        
        if (source == nameField) {
            statusLabel.setText("Enter your full name");
        } else if (source == ageField) {
            statusLabel.setText("Enter your age (numbers only)");
        } else if (source == emailField) {
            statusLabel.setText("Enter a valid email address");
        }
    }
    
    @Override
    public void focusLost(FocusEvent e) {
        TextField source = (TextField) e.getSource();
        source.setBackground(Color.WHITE); // Remove highlight
    }
    
    private void submitForm() {
        String name = nameField.getText().trim();
        String ageText = ageField.getText().trim();
        String email = emailField.getText().trim();
        
        StringBuilder result = new StringBuilder();
        result.append("=== FORM SUBMISSION ===\n");
        result.append("Name: ").append(name.isEmpty() ? "[Not provided]" : name).append("\n");
        
        // Validate and display age
        if (ageText.isEmpty()) {
            result.append("Age: [Not provided]\n");
        } else {
            try {
                int age = Integer.parseInt(ageText);
                result.append("Age: ").append(age).append(" years old\n");
            } catch (NumberFormatException e) {
                result.append("Age: [Invalid - not a number]\n");
            }
        }
        
        result.append("Email: ").append(email.isEmpty() ? "[Not provided]" : email).append("\n");
        
        // Simple email validation
        if (!email.isEmpty() && !email.contains("@")) {
            result.append("  Warning: Email format appears invalid\n");
        }
        
        result.append("Submitted at: ").append(new java.util.Date()).append("\n\n");
        
        resultArea.append(result.toString());
        statusLabel.setText("Form submitted successfully!");
    }
    
    private void clearForm() {
        nameField.setText("");
        ageField.setText("");
        emailField.setText("");
        resultArea.setText("Form cleared. Enter new information...\n");
        statusLabel.setText("All fields cleared");
        nameField.requestFocus();
    }
    
    private void validateAge() {
        String ageText = ageField.getText();
        try {
            if (!ageText.isEmpty()) {
                int age = Integer.parseInt(ageText);
                if (age < 0 || age > 150) {
                    ageField.setBackground(Color.PINK);
                } else {
                    ageField.setBackground(Color.YELLOW);
                }
            }
        } catch (NumberFormatException e) {
            ageField.setBackground(Color.PINK);
        }
    }
    
    private void validateEmail() {
        String email = emailField.getText();
        if (!email.isEmpty()) {
            if (email.contains("@") && email.contains(".")) {
                emailField.setBackground(Color.LIGHT_GREEN);
            } else {
                emailField.setBackground(Color.PINK);
            }
        } else {
            emailField.setBackground(Color.YELLOW);
        }
    }
    
    private void configureFrame() {
        setTitle("TextField Demo - User Registration");
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
        new TextFieldDemo();
    }
}
```

## 5. TextArea Component

TextArea provides multi-line text input and editing.

### TextArea Demo
```java
import java.awt.*;
import java.awt.event.*;

public class TextAreaDemo extends Frame implements ActionListener, TextListener {
    private TextArea textArea;
    private TextField searchField;
    private Button loadButton, saveButton, clearButton, searchButton;
    private Label statusLabel, statsLabel;
    private Checkbox wordWrapBox;
    
    public TextAreaDemo() {
        setupComponents();
        setupLayout();
        updateStats();
        configureFrame();
    }
    
    private void setupComponents() {
        // Text area with scrollbars
        textArea = new TextArea("Welcome to the AWT TextArea Demo!\n\n" +
                                "This is a multi-line text editor where you can:\n" +
                                "• Type multiple lines of text\n" +
                                "• Search for specific words\n" +
                                "• View real-time statistics\n" +
                                "• Toggle word wrapping\n\n" +
                                "Try typing some text and use the controls below!", 15, 50);
        textArea.setFont(new Font("Courier New", Font.PLAIN, 12));
        textArea.addTextListener(this);
        
        // Search field
        searchField = new TextField(20);
        searchField.addActionListener(this);
        
        // Buttons
        loadButton = new Button("Load Sample");
        saveButton = new Button("Save Content");
        clearButton = new Button("Clear All");
        searchButton = new Button("Search");
        
        loadButton.addActionListener(this);
        saveButton.addActionListener(this);
        clearButton.addActionListener(this);
        searchButton.addActionListener(this);
        
        // Checkbox for word wrap
        wordWrapBox = new Checkbox("Word Wrap", false);
        wordWrapBox.addItemListener(e -> {
            // Note: AWT TextArea doesn't have built-in word wrap control
            // This is just for demonstration
            statusLabel.setText("Word wrap toggled (feature limited in AWT)");
        });
        
        // Status and stats labels
        statusLabel = new Label("Ready - Start typing or use controls");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        statsLabel = new Label("");
        statsLabel.setFont(new Font("Arial", Font.PLAIN, 10));
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Top panel with controls
        Panel topPanel = new Panel(new FlowLayout());
        topPanel.add(loadButton);
        topPanel.add(saveButton);
        topPanel.add(clearButton);
        topPanel.add(new Label("  Search:"));
        topPanel.add(searchField);
        topPanel.add(searchButton);
        topPanel.add(wordWrapBox);
        
        // Bottom panel with status
        Panel bottomPanel = new Panel(new BorderLayout());
        bottomPanel.add(statusLabel, BorderLayout.CENTER);
        bottomPanel.add(statsLabel, BorderLayout.EAST);
        
        add(topPanel, BorderLayout.NORTH);
        add(textArea, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == loadButton) {
            loadSampleText();
        } else if (e.getSource() == saveButton) {
            saveContent();
        } else if (e.getSource() == clearButton) {
            clearContent();
        } else if (e.getSource() == searchButton || e.getSource() == searchField) {
            searchText();
        }
    }
    
    @Override
    public void textValueChanged(TextEvent e) {
        updateStats();
        statusLabel.setText("Text modified - " + new java.util.Date().toString());
    }
    
    private void loadSampleText() {
        String sampleText = "Sample Document\n" +
                           "==============\n\n" +
                           "This is a sample document loaded into the text area.\n\n" +
                           "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " +
                           "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " +
                           "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.\n\n" +
                           "Key Features:\n" +
                           "1. Multi-line text editing\n" +
                           "2. Real-time statistics\n" +
                           "3. Search functionality\n" +
                           "4. Content management\n\n" +
                           "You can modify this text as needed. " +
                           "The statistics will update automatically as you type.";
        
        textArea.setText(sampleText);
        statusLabel.setText("Sample text loaded successfully");
    }
    
    private void saveContent() {
        // In a real application, this would save to a file
        String content = textArea.getText();
        int charCount = content.length();
        
        statusLabel.setText("Content saved! (" + charCount + " characters saved)");
        
        // Simulate saving by showing content length
        System.out.println("Saving content:");
        System.out.println("Character count: " + charCount);
        System.out.println("Content preview: " + 
                          content.substring(0, Math.min(50, charCount)) + "...");
    }
    
    private void clearContent() {
        int option = showConfirmDialog("Are you sure you want to clear all content?");
        if (option == 1) { // Yes
            textArea.setText("");
            statusLabel.setText("Content cleared");
        }
    }
    
    private void searchText() {
        String searchTerm = searchField.getText().trim();
        if (searchTerm.isEmpty()) {
            statusLabel.setText("Please enter a search term");
            return;
        }
        
        String content = textArea.getText();
        String lowerContent = content.toLowerCase();
        String lowerSearch = searchTerm.toLowerCase();
        
        int index = lowerContent.indexOf(lowerSearch);
        if (index != -1) {
            // Found the text
            textArea.select(index, index + searchTerm.length());
            textArea.requestFocus();
            
            // Count occurrences
            int count = 0;
            int pos = 0;
            while ((pos = lowerContent.indexOf(lowerSearch, pos)) != -1) {
                count++;
                pos += lowerSearch.length();
            }
            
            statusLabel.setText("Found '" + searchTerm + "' - " + count + " occurrence(s)");
        } else {
            statusLabel.setText("'" + searchTerm + "' not found");
        }
    }
    
    private void updateStats() {
        String content = textArea.getText();
        int charCount = content.length();
        int lineCount = content.split("\n").length;
        int wordCount = content.trim().isEmpty() ? 0 : content.trim().split("\\s+").length;
        
        statsLabel.setText("Lines: " + lineCount + " | Words: " + wordCount + " | Chars: " + charCount);
    }
    
    private int showConfirmDialog(String message) {
        // Simple confirmation dialog simulation
        // In real AWT, you'd use Dialog class
        System.out.println("Confirm: " + message);
        return 1; // Assume "Yes" for demo
    }
    
    private void configureFrame() {
        setTitle("TextArea Demo - Multi-line Text Editor");
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
        new TextAreaDemo();
    }
}
```

## Summary

AWT Components provide the building blocks for creating functional user interfaces:

### Component Categories

| Component | Purpose | Key Features | Common Uses |
|-----------|---------|--------------|-------------|
| **Button** | Trigger actions | Click events, customizable appearance | Forms, toolbars, games |
| **Label** | Display text | Static text, alignment options | Instructions, titles, status |
| **TextField** | Single-line input | Text entry, validation, events | Forms, search boxes |
| **TextArea** | Multi-line input | Scrollable, large text editing | Editors, comments, logs |

### Event Handling Patterns

1. **ActionListener**: Button clicks, Enter key in text fields
2. **TextListener**: Real-time text changes
3. **FocusListener**: Component focus gained/lost
4. **ItemListener**: Checkbox and choice selections

### Best Practices

1. **Event Management**: Use appropriate listeners for different interactions
2. **Validation**: Provide real-time feedback for user input
3. **Visual Feedback**: Use colors and fonts to indicate status
4. **Accessibility**: Ensure components are keyboard navigable
5. **User Experience**: Provide clear instructions and status updates

### Common Patterns

- **Form Validation**: Real-time checking with visual indicators
- **Search Functionality**: Text field with search button and highlighting
- **Dynamic Content**: Labels and text areas that update based on user actions
- **Interactive Elements**: Buttons that change appearance and behavior

These fundamental components form the foundation for more complex AWT applications and provide essential user interaction capabilities.

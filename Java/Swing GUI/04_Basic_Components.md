# Basic Swing Components

## 1. JLabel - Displaying Text and Images

JLabel is used to display read-only text, images, or both.

### Basic JLabel Examples
```java
import javax.swing.*;
import java.awt.*;
import java.net.URL;

public class JLabelDemo extends JFrame {
    
    public JLabelDemo() {
        setupLabels();
        configureFrame();
    }
    
    private void setupLabels() {
        setLayout(new GridLayout(5, 2, 10, 10));
        
        // Basic text label
        JLabel basicLabel = new JLabel("Basic Text Label");
        add(new JLabel("Basic Label:"));
        add(basicLabel);
        
        // Label with HTML formatting
        JLabel htmlLabel = new JLabel("<html><b>Bold</b>, <i>Italic</i>, <font color='red'>Red Text</font></html>");
        add(new JLabel("HTML Label:"));
        add(htmlLabel);
        
        // Label with different alignments
        JLabel centerLabel = new JLabel("Centered Text", JLabel.CENTER);
        centerLabel.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        add(new JLabel("Center Aligned:"));
        add(centerLabel);
        
        // Label with custom font
        JLabel fontLabel = new JLabel("Custom Font");
        fontLabel.setFont(new Font("Arial", Font.BOLD, 16));
        fontLabel.setForeground(Color.BLUE);
        add(new JLabel("Custom Font:"));
        add(fontLabel);
        
        // Label with icon (using a simple colored rectangle as icon)
        JLabel iconLabel = new JLabel("Text with Icon", createSimpleIcon(Color.GREEN), JLabel.LEFT);
        iconLabel.setHorizontalTextPosition(JLabel.RIGHT);
        iconLabel.setVerticalTextPosition(JLabel.CENTER);
        add(new JLabel("With Icon:"));
        add(iconLabel);
    }
    
    // Helper method to create a simple colored icon
    private Icon createSimpleIcon(Color color) {
        return new Icon() {
            @Override
            public void paintIcon(Component c, Graphics g, int x, int y) {
                g.setColor(color);
                g.fillRect(x, y, getIconWidth(), getIconHeight());
                g.setColor(Color.BLACK);
                g.drawRect(x, y, getIconWidth() - 1, getIconHeight() - 1);
            }
            
            @Override
            public int getIconWidth() { return 16; }
            
            @Override
            public int getIconHeight() { return 16; }
        };
    }
    
    private void configureFrame() {
        setTitle("JLabel Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 300);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JLabelDemo().setVisible(true);
        });
    }
}
```

### Advanced JLabel Features
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class AdvancedJLabelDemo extends JFrame {
    
    public AdvancedJLabelDemo() {
        setupAdvancedLabels();
        configureFrame();
    }
    
    private void setupAdvancedLabels() {
        setLayout(new BorderLayout());
        
        // Multi-line label with HTML
        JLabel multiLineLabel = new JLabel(
            "<html>" +
            "<h2>Product Information</h2>" +
            "<p><b>Name:</b> Super Widget 3000</p>" +
            "<p><b>Price:</b> <font color='green'>$99.99</font></p>" +
            "<p><b>Status:</b> <font color='red'>In Stock</font></p>" +
            "<p><i>Click for details...</i></p>" +
            "</html>"
        );
        multiLineLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Product Card"),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        multiLineLabel.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        
        // Add click handler to make it interactive
        multiLineLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                JOptionPane.showMessageDialog(
                    AdvancedJLabelDemo.this,
                    "Product details would be shown here!",
                    "Product Details",
                    JOptionPane.INFORMATION_MESSAGE
                );
            }
            
            @Override
            public void mouseEntered(MouseEvent e) {
                multiLineLabel.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createTitledBorder("Product Card (Click me!)"),
                    BorderFactory.createEmptyBorder(10, 10, 10, 10)
                ));
            }
            
            @Override
            public void mouseExited(MouseEvent e) {
                multiLineLabel.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createTitledBorder("Product Card"),
                    BorderFactory.createEmptyBorder(10, 10, 10, 10)
                ));
            }
        });
        
        // Status bar label
        JLabel statusLabel = new JLabel("Status: Ready");
        statusLabel.setBorder(BorderFactory.createLoweredBevelBorder());
        statusLabel.setOpaque(true);
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        // Dynamic updating label
        JLabel timeLabel = new JLabel("Current Time: Loading...");
        Timer timer = new Timer(1000, e -> {
            timeLabel.setText("Current Time: " + java.time.LocalTime.now().toString());
        });
        timer.start();
        
        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.add(timeLabel, BorderLayout.CENTER);
        bottomPanel.add(statusLabel, BorderLayout.SOUTH);
        
        add(multiLineLabel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }
    
    private void configureFrame() {
        setTitle("Advanced JLabel Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 300);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new AdvancedJLabelDemo().setVisible(true);
        });
    }
}
```

## 2. JButton - Interactive Buttons

JButton is used to trigger actions when clicked.

### Basic JButton Examples
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class JButtonDemo extends JFrame {
    private JLabel resultLabel;
    private int clickCount = 0;
    
    public JButtonDemo() {
        setupButtons();
        configureFrame();
    }
    
    private void setupButtons() {
        setLayout(new BorderLayout());
        
        resultLabel = new JLabel("Click any button to see result", JLabel.CENTER);
        resultLabel.setFont(new Font("Arial", Font.PLAIN, 14));
        resultLabel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        JPanel buttonPanel = new JPanel(new GridLayout(3, 2, 10, 10));
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        // Basic button
        JButton basicButton = new JButton("Basic Button");
        basicButton.addActionListener(e -> resultLabel.setText("Basic button clicked!"));
        
        // Button with icon
        JButton iconButton = new JButton("With Icon", createColorIcon(Color.BLUE));
        iconButton.addActionListener(e -> resultLabel.setText("Icon button clicked!"));
        
        // Disabled button
        JButton disabledButton = new JButton("Disabled Button");
        disabledButton.setEnabled(false);
        disabledButton.setToolTipText("This button is currently disabled");
        
        // Toggle button
        JButton toggleButton = new JButton("Enable/Disable");
        toggleButton.addActionListener(e -> {
            disabledButton.setEnabled(!disabledButton.isEnabled());
            String status = disabledButton.isEnabled() ? "enabled" : "disabled";
            resultLabel.setText("Disabled button is now " + status);
        });
        
        // Counter button
        JButton counterButton = new JButton("Click Counter: 0");
        counterButton.addActionListener(e -> {
            clickCount++;
            counterButton.setText("Click Counter: " + clickCount);
            resultLabel.setText("Counter updated to: " + clickCount);
        });
        
        // Color changing button
        JButton colorButton = new JButton("Color Changer");
        colorButton.addActionListener(e -> {
            Color[] colors = {Color.RED, Color.GREEN, Color.BLUE, Color.ORANGE, Color.PINK};
            Color randomColor = colors[(int) (Math.random() * colors.length)];
            resultLabel.setForeground(randomColor);
            resultLabel.setText("Text color changed to " + getColorName(randomColor));
        });
        
        buttonPanel.add(basicButton);
        buttonPanel.add(iconButton);
        buttonPanel.add(disabledButton);
        buttonPanel.add(toggleButton);
        buttonPanel.add(counterButton);
        buttonPanel.add(colorButton);
        
        add(resultLabel, BorderLayout.NORTH);
        add(buttonPanel, BorderLayout.CENTER);
    }
    
    private Icon createColorIcon(Color color) {
        return new Icon() {
            @Override
            public void paintIcon(Component c, Graphics g, int x, int y) {
                g.setColor(color);
                g.fillOval(x, y, getIconWidth(), getIconHeight());
                g.setColor(Color.BLACK);
                g.drawOval(x, y, getIconWidth() - 1, getIconHeight() - 1);
            }
            
            @Override
            public int getIconWidth() { return 12; }
            
            @Override
            public int getIconHeight() { return 12; }
        };
    }
    
    private String getColorName(Color color) {
        if (color.equals(Color.RED)) return "Red";
        if (color.equals(Color.GREEN)) return "Green";
        if (color.equals(Color.BLUE)) return "Blue";
        if (color.equals(Color.ORANGE)) return "Orange";
        if (color.equals(Color.PINK)) return "Pink";
        return "Unknown";
    }
    
    private void configureFrame() {
        setTitle("JButton Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(500, 300);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JButtonDemo().setVisible(true);
        });
    }
}
```

### Advanced JButton Features
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class AdvancedJButtonDemo extends JFrame {
    
    public AdvancedJButtonDemo() {
        setupAdvancedButtons();
        configureFrame();
    }
    
    private void setupAdvancedButtons() {
        setLayout(new GridLayout(2, 2, 10, 10));
        
        // Custom styled button
        JButton styledButton = new JButton("Styled Button");
        styledButton.setFont(new Font("Arial", Font.BOLD, 14));
        styledButton.setBackground(Color.BLUE);
        styledButton.setForeground(Color.WHITE);
        styledButton.setFocusPainted(false);
        styledButton.setBorder(BorderFactory.createRaisedBevelBorder());
        styledButton.addActionListener(e -> 
            JOptionPane.showMessageDialog(this, "Styled button clicked!"));
        
        // Button with keyboard shortcut (mnemonic)
        JButton mnemonicButton = new JButton("Press Alt+S");
        mnemonicButton.setMnemonic('S');
        mnemonicButton.setToolTipText("You can press Alt+S to activate this button");
        mnemonicButton.addActionListener(e -> 
            JOptionPane.showMessageDialog(this, "Shortcut worked!"));
        
        // Rollover button
        JButton rolloverButton = new JButton("Hover Effects");
        rolloverButton.setRolloverEnabled(true);
        rolloverButton.addChangeListener(e -> {
            ButtonModel model = rolloverButton.getModel();
            if (model.isRollover()) {
                rolloverButton.setBackground(Color.YELLOW);
            } else {
                rolloverButton.setBackground(UIManager.getColor("Button.background"));
            }
        });
        rolloverButton.addActionListener(e -> 
            JOptionPane.showMessageDialog(this, "Rollover button clicked!"));
        
        // Multi-action button
        JButton multiActionButton = new JButton("Multi-Action (Right-click too)");
        multiActionButton.addActionListener(e -> 
            JOptionPane.showMessageDialog(this, "Left-clicked!"));
        
        multiActionButton.addMouseListener(new java.awt.event.MouseAdapter() {
            @Override
            public void mouseClicked(java.awt.event.MouseEvent e) {
                if (SwingUtilities.isRightMouseButton(e)) {
                    JPopupMenu popup = new JPopupMenu();
                    popup.add(new JMenuItem("Right-click Option 1"));
                    popup.add(new JMenuItem("Right-click Option 2"));
                    popup.show(e.getComponent(), e.getX(), e.getY());
                }
            }
        });
        
        add(styledButton);
        add(mnemonicButton);
        add(rolloverButton);
        add(multiActionButton);
    }
    
    private void configureFrame() {
        setTitle("Advanced JButton Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(500, 300);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new AdvancedJButtonDemo().setVisible(true);
        });
    }
}
```

## 3. JTextField - Single-line Text Input

JTextField allows users to input and edit a single line of text.

### Basic JTextField Examples
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;

public class JTextFieldDemo extends JFrame {
    private JTextField nameField, emailField, phoneField, passwordField;
    private JLabel resultLabel;
    
    public JTextFieldDemo() {
        setupTextFields();
        configureFrame();
    }
    
    private void setupTextFields() {
        setLayout(new BorderLayout());
        
        // Create form panel
        JPanel formPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        
        // Name field
        gbc.gridx = 0; gbc.gridy = 0;
        formPanel.add(new JLabel("Name:"), gbc);
        gbc.gridx = 1;
        nameField = new JTextField(20);
        nameField.setToolTipText("Enter your full name");
        formPanel.add(nameField, gbc);
        
        // Email field with validation
        gbc.gridx = 0; gbc.gridy = 1;
        formPanel.add(new JLabel("Email:"), gbc);
        gbc.gridx = 1;
        emailField = new JTextField(20);
        emailField.setToolTipText("Enter your email address");
        emailField.addFocusListener(new FocusAdapter() {
            @Override
            public void focusLost(FocusEvent e) {
                validateEmail();
            }
        });
        formPanel.add(emailField, gbc);
        
        // Phone field with input formatting
        gbc.gridx = 0; gbc.gridy = 2;
        formPanel.add(new JLabel("Phone:"), gbc);
        gbc.gridx = 1;
        phoneField = new JTextField(20);
        phoneField.setToolTipText("Enter phone number (numbers only)");
        // Add document filter to allow only numbers
        phoneField.addKeyListener(new java.awt.event.KeyAdapter() {
            @Override
            public void keyTyped(java.awt.event.KeyEvent e) {
                char c = e.getKeyChar();
                if (!Character.isDigit(c) && c != '-' && c != '(' && c != ')' && c != ' ') {
                    e.consume(); // Ignore non-numeric characters
                }
            }
        });
        formPanel.add(phoneField, gbc);
        
        // Password field
        gbc.gridx = 0; gbc.gridy = 3;
        formPanel.add(new JLabel("Password:"), gbc);
        gbc.gridx = 1;
        passwordField = new JPasswordField(20);
        passwordField.setToolTipText("Enter password (hidden text)");
        formPanel.add(passwordField, gbc);
        
        // Buttons
        gbc.gridx = 0; gbc.gridy = 4;
        gbc.gridwidth = 2;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        
        JPanel buttonPanel = new JPanel(new FlowLayout());
        JButton submitButton = new JButton("Submit");
        JButton clearButton = new JButton("Clear");
        JButton showPasswordButton = new JButton("Show/Hide Password");
        
        submitButton.addActionListener(e -> handleSubmit());
        clearButton.addActionListener(e -> clearAllFields());
        showPasswordButton.addActionListener(e -> togglePasswordVisibility());
        
        buttonPanel.add(submitButton);
        buttonPanel.add(clearButton);
        buttonPanel.add(showPasswordButton);
        formPanel.add(buttonPanel, gbc);
        
        // Result label
        resultLabel = new JLabel("Fill in the form and click Submit");
        resultLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Result"),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        
        add(formPanel, BorderLayout.CENTER);
        add(resultLabel, BorderLayout.SOUTH);
    }
    
    private void validateEmail() {
        String email = emailField.getText().trim();
        if (!email.isEmpty() && !email.contains("@")) {
            emailField.setBorder(BorderFactory.createLineBorder(Color.RED, 2));
            resultLabel.setText("Invalid email format");
        } else {
            emailField.setBorder(UIManager.getBorder("TextField.border"));
            if (!email.isEmpty()) {
                resultLabel.setText("Email format looks good");
            }
        }
    }
    
    private void handleSubmit() {
        StringBuilder result = new StringBuilder("Form Data:<br>");
        result.append("Name: ").append(nameField.getText()).append("<br>");
        result.append("Email: ").append(emailField.getText()).append("<br>");
        result.append("Phone: ").append(phoneField.getText()).append("<br>");
        result.append("Password: ").append("*".repeat(passwordField.getPassword().length));
        
        resultLabel.setText("<html>" + result.toString() + "</html>");
    }
    
    private void clearAllFields() {
        nameField.setText("");
        emailField.setText("");
        phoneField.setText("");
        passwordField.setText("");
        resultLabel.setText("All fields cleared");
        nameField.requestFocus();
    }
    
    private void togglePasswordVisibility() {
        if (passwordField.getEchoChar() == 0) {
            passwordField.setEchoChar('*');
            resultLabel.setText("Password hidden");
        } else {
            passwordField.setEchoChar((char) 0);
            resultLabel.setText("Password visible");
        }
    }
    
    private void configureFrame() {
        setTitle("JTextField Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JTextFieldDemo().setVisible(true);
        });
    }
}
```

## 4. JTextArea - Multi-line Text Input

JTextArea allows users to input and edit multiple lines of text.

### JTextArea Examples
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

public class JTextAreaDemo extends JFrame {
    private JTextArea textArea;
    private JLabel statusLabel;
    private JLabel statsLabel;
    
    public JTextAreaDemo() {
        setupTextArea();
        configureFrame();
    }
    
    private void setupTextArea() {
        setLayout(new BorderLayout());
        
        // Main text area
        textArea = new JTextArea(15, 40);
        textArea.setFont(new Font("Courier New", Font.PLAIN, 14));
        textArea.setLineWrap(true);
        textArea.setWrapStyleWord(true);
        textArea.setText("Welcome to the text editor!\n\nType your content here...\n");
        
        // Add document listener for real-time statistics
        textArea.getDocument().addDocumentListener(new DocumentListener() {
            public void insertUpdate(DocumentEvent e) { updateStats(); }
            public void removeUpdate(DocumentEvent e) { updateStats(); }
            public void changedUpdate(DocumentEvent e) { updateStats(); }
        });
        
        JScrollPane scrollPane = new JScrollPane(textArea);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        
        // Button panel
        JPanel buttonPanel = new JPanel(new FlowLayout());
        
        JButton clearButton = new JButton("Clear");
        JButton upperCaseButton = new JButton("UPPERCASE");
        JButton lowerCaseButton = new JButton("lowercase");
        JButton reverseButton = new JButton("Reverse");
        JButton insertTimeButton = new JButton("Insert Time");
        JButton wordWrapButton = new JButton("Toggle Word Wrap");
        
        clearButton.addActionListener(e -> {
            textArea.setText("");
            statusLabel.setText("Text cleared");
        });
        
        upperCaseButton.addActionListener(e -> {
            String selected = textArea.getSelectedText();
            if (selected != null) {
                textArea.replaceSelection(selected.toUpperCase());
                statusLabel.setText("Selected text converted to uppercase");
            } else {
                textArea.setText(textArea.getText().toUpperCase());
                statusLabel.setText("All text converted to uppercase");
            }
        });
        
        lowerCaseButton.addActionListener(e -> {
            String selected = textArea.getSelectedText();
            if (selected != null) {
                textArea.replaceSelection(selected.toLowerCase());
                statusLabel.setText("Selected text converted to lowercase");
            } else {
                textArea.setText(textArea.getText().toLowerCase());
                statusLabel.setText("All text converted to lowercase");
            }
        });
        
        reverseButton.addActionListener(e -> {
            String text = textArea.getText();
            String reversed = new StringBuilder(text).reverse().toString();
            textArea.setText(reversed);
            statusLabel.setText("Text reversed");
        });
        
        insertTimeButton.addActionListener(e -> {
            String timestamp = java.time.LocalDateTime.now().toString();
            textArea.insert("\\n[" + timestamp + "]\\n", textArea.getCaretPosition());
            statusLabel.setText("Timestamp inserted");
        });
        
        wordWrapButton.addActionListener(e -> {
            textArea.setLineWrap(!textArea.getLineWrap());
            textArea.setWrapStyleWord(textArea.getLineWrap());
            statusLabel.setText("Word wrap " + (textArea.getLineWrap() ? "enabled" : "disabled"));
        });
        
        buttonPanel.add(clearButton);
        buttonPanel.add(upperCaseButton);
        buttonPanel.add(lowerCaseButton);
        buttonPanel.add(reverseButton);
        buttonPanel.add(insertTimeButton);
        buttonPanel.add(wordWrapButton);
        
        // Status panel
        JPanel statusPanel = new JPanel(new BorderLayout());
        statusLabel = new JLabel("Ready");
        statsLabel = new JLabel("Characters: 0, Words: 0, Lines: 0");
        
        statusPanel.add(statusLabel, BorderLayout.WEST);
        statusPanel.add(statsLabel, BorderLayout.EAST);
        statusPanel.setBorder(BorderFactory.createLoweredBevelBorder());
        
        add(scrollPane, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.NORTH);
        add(statusPanel, BorderLayout.SOUTH);
        
        // Initial stats update
        updateStats();
    }
    
    private void updateStats() {
        String text = textArea.getText();
        int characters = text.length();
        int words = text.trim().isEmpty() ? 0 : text.trim().split("\\s+").length;
        int lines = text.split("\n").length;
        
        statsLabel.setText(String.format("Characters: %d, Words: %d, Lines: %d", 
                                        characters, words, lines));
    }
    
    private void configureFrame() {
        setTitle("JTextArea Demo - Text Editor");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JTextAreaDemo().setVisible(true);
        });
    }
}
```

## 5. JCheckBox - Boolean Selection

JCheckBox allows users to make boolean (yes/no) selections.

### JCheckBox Examples
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.ArrayList;
import java.util.List;

public class JCheckBoxDemo extends JFrame {
    private List<JCheckBox> featureCheckBoxes;
    private JCheckBox selectAllCheckBox;
    private JLabel resultLabel;
    private JLabel priceLabel;
    
    public JCheckBoxDemo() {
        setupCheckBoxes();
        configureFrame();
    }
    
    private void setupCheckBoxes() {
        setLayout(new BorderLayout());
        
        // Main panel
        JPanel mainPanel = new JPanel(new BorderLayout());
        
        // Features panel
        JPanel featuresPanel = new JPanel();
        featuresPanel.setLayout(new BoxLayout(featuresPanel, BoxLayout.Y_AXIS));
        featuresPanel.setBorder(BorderFactory.createTitledBorder("Select Software Features"));
        
        // Feature options with prices
        String[] features = {
            "Basic Package - $10",
            "Advanced Graphics - $25",
            "Database Support - $15",
            "Cloud Integration - $20",
            "Mobile Sync - $12",
            "Premium Support - $30"
        };
        
        int[] prices = {10, 25, 15, 20, 12, 30};
        
        featureCheckBoxes = new ArrayList<>();
        
        for (int i = 0; i < features.length; i++) {
            JCheckBox checkBox = new JCheckBox(features[i]);
            checkBox.putClientProperty("price", prices[i]);
            checkBox.addItemListener(new FeatureItemListener());
            featureCheckBoxes.add(checkBox);
            featuresPanel.add(checkBox);
        }
        
        // Select All checkbox
        selectAllCheckBox = new JCheckBox("Select All Features");
        selectAllCheckBox.setFont(new Font("Arial", Font.BOLD, 12));
        selectAllCheckBox.addItemListener(e -> {
            boolean selected = selectAllCheckBox.isSelected();
            for (JCheckBox checkBox : featureCheckBoxes) {
                checkBox.setSelected(selected);
            }
        });
        
        JPanel selectAllPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        selectAllPanel.add(selectAllCheckBox);
        
        // Options panel
        JPanel optionsPanel = new JPanel();
        optionsPanel.setLayout(new BoxLayout(optionsPanel, BoxLayout.Y_AXIS));
        optionsPanel.setBorder(BorderFactory.createTitledBorder("Additional Options"));
        
        JCheckBox emailUpdatesCheckBox = new JCheckBox("Receive email updates", true);
        JCheckBox betaFeaturesCheckBox = new JCheckBox("Enable beta features");
        JCheckBox analyticsCheckBox = new JCheckBox("Send usage analytics");
        
        emailUpdatesCheckBox.addItemListener(e -> updateResult());
        betaFeaturesCheckBox.addItemListener(e -> {
            if (betaFeaturesCheckBox.isSelected()) {
                int result = JOptionPane.showConfirmDialog(
                    this,
                    "Beta features may be unstable. Continue?",
                    "Warning",
                    JOptionPane.YES_NO_OPTION,
                    JOptionPane.WARNING_MESSAGE
                );
                if (result != JOptionPane.YES_OPTION) {
                    betaFeaturesCheckBox.setSelected(false);
                }
            }
            updateResult();
        });
        analyticsCheckBox.addItemListener(e -> updateResult());
        
        optionsPanel.add(emailUpdatesCheckBox);
        optionsPanel.add(betaFeaturesCheckBox);
        optionsPanel.add(analyticsCheckBox);
        
        // Result and price labels
        resultLabel = new JLabel("No features selected");
        resultLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Selected Features"),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        
        priceLabel = new JLabel("Total Price: $0");
        priceLabel.setFont(new Font("Arial", Font.BOLD, 16));
        priceLabel.setForeground(Color.BLUE);
        priceLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Layout
        JPanel topPanel = new JPanel(new BorderLayout());
        topPanel.add(selectAllPanel, BorderLayout.NORTH);
        topPanel.add(featuresPanel, BorderLayout.CENTER);
        
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.add(topPanel, BorderLayout.NORTH);
        centerPanel.add(optionsPanel, BorderLayout.CENTER);
        
        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.add(resultLabel, BorderLayout.CENTER);
        bottomPanel.add(priceLabel, BorderLayout.SOUTH);
        
        add(centerPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
        
        updateResult();
    }
    
    private class FeatureItemListener implements ItemListener {
        @Override
        public void itemStateChanged(ItemEvent e) {
            updateResult();
            updateSelectAllState();
        }
    }
    
    private void updateResult() {
        List<String> selectedFeatures = new ArrayList<>();
        int totalPrice = 0;
        
        for (JCheckBox checkBox : featureCheckBoxes) {
            if (checkBox.isSelected()) {
                selectedFeatures.add(checkBox.getText());
                totalPrice += (Integer) checkBox.getClientProperty("price");
            }
        }
        
        if (selectedFeatures.isEmpty()) {
            resultLabel.setText("No features selected");
        } else {
            StringBuilder html = new StringBuilder("<html>");
            for (String feature : selectedFeatures) {
                html.append("• ").append(feature).append("<br>");
            }
            html.append("</html>");
            resultLabel.setText(html.toString());
        }
        
        priceLabel.setText("Total Price: $" + totalPrice);
        
        // Color code the price
        if (totalPrice == 0) {
            priceLabel.setForeground(Color.GRAY);
        } else if (totalPrice <= 50) {
            priceLabel.setForeground(Color.GREEN);
        } else if (totalPrice <= 100) {
            priceLabel.setForeground(Color.ORANGE);
        } else {
            priceLabel.setForeground(Color.RED);
        }
    }
    
    private void updateSelectAllState() {
        int selectedCount = 0;
        for (JCheckBox checkBox : featureCheckBoxes) {
            if (checkBox.isSelected()) {
                selectedCount++;
            }
        }
        
        if (selectedCount == 0) {
            selectAllCheckBox.setSelected(false);
            selectAllCheckBox.setText("Select All Features");
        } else if (selectedCount == featureCheckBoxes.size()) {
            selectAllCheckBox.setSelected(true);
            selectAllCheckBox.setText("Deselect All Features");
        } else {
            selectAllCheckBox.setSelected(false);
            selectAllCheckBox.setText("Select All Features (" + selectedCount + "/" + featureCheckBoxes.size() + ")");
        }
    }
    
    private void configureFrame() {
        setTitle("JCheckBox Demo - Software Package Selector");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JCheckBoxDemo().setVisible(true);
        });
    }
}
```

## 6. JRadioButton - Exclusive Selection

JRadioButton allows users to select one option from a group of mutually exclusive options.

### JRadioButton Examples
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class JRadioButtonDemo extends JFrame {
    private ButtonGroup sizeGroup, crustGroup, deliveryGroup;
    private JLabel orderSummaryLabel;
    private JLabel priceLabel;
    
    public JRadioButtonDemo() {
        setupRadioButtons();
        configureFrame();
    }
    
    private void setupRadioButtons() {
        setLayout(new BorderLayout());
        
        JPanel mainPanel = new JPanel(new GridLayout(1, 3, 10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        // Pizza Size Selection
        JPanel sizePanel = new JPanel();
        sizePanel.setLayout(new BoxLayout(sizePanel, BoxLayout.Y_AXIS));
        sizePanel.setBorder(BorderFactory.createTitledBorder("Pizza Size"));
        
        sizeGroup = new ButtonGroup();
        JRadioButton smallRadio = new JRadioButton("Small - $12", true);
        JRadioButton mediumRadio = new JRadioButton("Medium - $15");
        JRadioButton largeRadio = new JRadioButton("Large - $18");
        JRadioButton extraLargeRadio = new JRadioButton("Extra Large - $22");
        
        smallRadio.setActionCommand("Small,12");
        mediumRadio.setActionCommand("Medium,15");
        largeRadio.setActionCommand("Large,18");
        extraLargeRadio.setActionCommand("Extra Large,22");
        
        sizeGroup.add(smallRadio);
        sizeGroup.add(mediumRadio);
        sizeGroup.add(largeRadio);
        sizeGroup.add(extraLargeRadio);
        
        ActionListener sizeListener = e -> updateOrderSummary();
        smallRadio.addActionListener(sizeListener);
        mediumRadio.addActionListener(sizeListener);
        largeRadio.addActionListener(sizeListener);
        extraLargeRadio.addActionListener(sizeListener);
        
        sizePanel.add(smallRadio);
        sizePanel.add(mediumRadio);
        sizePanel.add(largeRadio);
        sizePanel.add(extraLargeRadio);
        
        // Crust Type Selection
        JPanel crustPanel = new JPanel();
        crustPanel.setLayout(new BoxLayout(crustPanel, BoxLayout.Y_AXIS));
        crustPanel.setBorder(BorderFactory.createTitledBorder("Crust Type"));
        
        crustGroup = new ButtonGroup();
        JRadioButton thinRadio = new JRadioButton("Thin Crust", true);
        JRadioButton thickRadio = new JRadioButton("Thick Crust");
        JRadioButton stuffedRadio = new JRadioButton("Stuffed Crust (+$3)");
        JRadioButton glutenFreeRadio = new JRadioButton("Gluten Free (+$2)");
        
        thinRadio.setActionCommand("Thin Crust,0");
        thickRadio.setActionCommand("Thick Crust,0");
        stuffedRadio.setActionCommand("Stuffed Crust,3");
        glutenFreeRadio.setActionCommand("Gluten Free,2");
        
        crustGroup.add(thinRadio);
        crustGroup.add(thickRadio);
        crustGroup.add(stuffedRadio);
        crustGroup.add(glutenFreeRadio);
        
        ActionListener crustListener = e -> updateOrderSummary();
        thinRadio.addActionListener(crustListener);
        thickRadio.addActionListener(crustListener);
        stuffedRadio.addActionListener(crustListener);
        glutenFreeRadio.addActionListener(crustListener);
        
        crustPanel.add(thinRadio);
        crustPanel.add(thickRadio);
        crustPanel.add(stuffedRadio);
        crustPanel.add(glutenFreeRadio);
        
        // Delivery Option Selection
        JPanel deliveryPanel = new JPanel();
        deliveryPanel.setLayout(new BoxLayout(deliveryPanel, BoxLayout.Y_AXIS));
        deliveryPanel.setBorder(BorderFactory.createTitledBorder("Delivery Option"));
        
        deliveryGroup = new ButtonGroup();
        JRadioButton pickupRadio = new JRadioButton("Pickup (Free)", true);
        JRadioButton standardRadio = new JRadioButton("Standard Delivery (+$3)");
        JRadioButton expressRadio = new JRadioButton("Express Delivery (+$6)");
        
        pickupRadio.setActionCommand("Pickup,0");
        standardRadio.setActionCommand("Standard Delivery,3");
        expressRadio.setActionCommand("Express Delivery,6");
        
        deliveryGroup.add(pickupRadio);
        deliveryGroup.add(standardRadio);
        deliveryGroup.add(expressRadio);
        
        ActionListener deliveryListener = e -> updateOrderSummary();
        pickupRadio.addActionListener(deliveryListener);
        standardRadio.addActionListener(deliveryListener);
        expressRadio.addActionListener(deliveryListener);
        
        deliveryPanel.add(pickupRadio);
        deliveryPanel.add(standardRadio);
        deliveryPanel.add(expressRadio);
        
        mainPanel.add(sizePanel);
        mainPanel.add(crustPanel);
        mainPanel.add(deliveryPanel);
        
        // Order summary and controls
        JPanel bottomPanel = new JPanel(new BorderLayout());
        
        orderSummaryLabel = new JLabel();
        orderSummaryLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Order Summary"),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        
        priceLabel = new JLabel();
        priceLabel.setFont(new Font("Arial", Font.BOLD, 18));
        priceLabel.setHorizontalAlignment(SwingConstants.CENTER);
        priceLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        JPanel buttonPanel = new JPanel(new FlowLayout());
        JButton orderButton = new JButton("Place Order");
        JButton resetButton = new JButton("Reset Order");
        
        orderButton.addActionListener(e -> {
            JOptionPane.showMessageDialog(this, 
                "Order placed successfully!\n" + getCurrentOrder(),
                "Order Confirmation",
                JOptionPane.INFORMATION_MESSAGE);
        });
        
        resetButton.addActionListener(e -> resetToDefaults());
        
        buttonPanel.add(orderButton);
        buttonPanel.add(resetButton);
        
        bottomPanel.add(orderSummaryLabel, BorderLayout.CENTER);
        bottomPanel.add(priceLabel, BorderLayout.EAST);
        bottomPanel.add(buttonPanel, BorderLayout.SOUTH);
        
        add(mainPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
        
        updateOrderSummary();
    }
    
    private void updateOrderSummary() {
        String order = getCurrentOrder();
        orderSummaryLabel.setText("<html>" + order.replace("\n", "<br>") + "</html>");
        
        int totalPrice = calculateTotalPrice();
        priceLabel.setText("Total: $" + totalPrice);
        
        // Color code the price
        if (totalPrice <= 15) {
            priceLabel.setForeground(Color.GREEN);
        } else if (totalPrice <= 25) {
            priceLabel.setForeground(Color.ORANGE);
        } else {
            priceLabel.setForeground(Color.RED);
        }
    }
    
    private String getCurrentOrder() {
        StringBuilder order = new StringBuilder();
        
        String sizeSelection = sizeGroup.getSelection().getActionCommand();
        String[] sizeParts = sizeSelection.split(",");
        order.append("Size: ").append(sizeParts[0]).append("\n");
        
        String crustSelection = crustGroup.getSelection().getActionCommand();
        String[] crustParts = crustSelection.split(",");
        order.append("Crust: ").append(crustParts[0]).append("\n");
        
        String deliverySelection = deliveryGroup.getSelection().getActionCommand();
        String[] deliveryParts = deliverySelection.split(",");
        order.append("Delivery: ").append(deliveryParts[0]);
        
        return order.toString();
    }
    
    private int calculateTotalPrice() {
        int total = 0;
        
        String sizeSelection = sizeGroup.getSelection().getActionCommand();
        total += Integer.parseInt(sizeSelection.split(",")[1]);
        
        String crustSelection = crustGroup.getSelection().getActionCommand();
        total += Integer.parseInt(crustSelection.split(",")[1]);
        
        String deliverySelection = deliveryGroup.getSelection().getActionCommand();
        total += Integer.parseInt(deliverySelection.split(",")[1]);
        
        return total;
    }
    
    private void resetToDefaults() {
        // Reset to first option in each group
        ((JRadioButton) sizeGroup.getElements().nextElement()).setSelected(true);
        ((JRadioButton) crustGroup.getElements().nextElement()).setSelected(true);
        ((JRadioButton) deliveryGroup.getElements().nextElement()).setSelected(true);
        
        updateOrderSummary();
    }
    
    private void configureFrame() {
        setTitle("JRadioButton Demo - Pizza Order System");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JRadioButtonDemo().setVisible(true);
        });
    }
}
```

## Summary

Basic Swing components provide the foundation for user interface interaction:

### Component Overview
- **JLabel**: Display text, images, or HTML content
- **JButton**: Trigger actions and events
- **JTextField**: Single-line text input with validation
- **JTextArea**: Multi-line text editing with advanced features
- **JCheckBox**: Boolean selections and feature toggles
- **JRadioButton**: Mutually exclusive choices with ButtonGroup

### Key Features Demonstrated
- Event handling for user interactions
- Input validation and formatting
- Real-time updates and feedback
- Custom styling and appearance
- Accessibility features (mnemonics, tooltips)
- HTML rendering for rich text display

### Best Practices
- Provide immediate visual feedback
- Use appropriate input validation
- Group related radio buttons with ButtonGroup
- Add tooltips for user guidance
- Implement keyboard shortcuts where appropriate
- Use consistent styling throughout the application
- Handle edge cases and invalid input gracefully

### Common Patterns
- Form validation on focus lost
- Real-time statistics and updates
- Toggle functionality for settings
- Grouped selections with summary displays
- Custom styling for enhanced user experience

These basic components form the building blocks for more complex Swing applications. In the next sections, we'll explore advanced components and container management.

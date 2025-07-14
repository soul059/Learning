# Introduction to Swing GUI Programming

## 1. What is Swing?

Swing is Java's powerful GUI (Graphical User Interface) toolkit that provides a rich set of lightweight components for building desktop applications. It's part of the Java Foundation Classes (JFC) and offers platform-independent GUI components.

### Key Features of Swing
- **Lightweight Components**: Swing components are written entirely in Java
- **Platform Independent**: Look and feel can be customized
- **Rich Component Set**: Extensive collection of GUI components
- **MVC Architecture**: Model-View-Controller design pattern
- **Pluggable Look and Feel**: Can mimic native OS appearance

## 2. Swing vs AWT

| Feature | AWT | Swing |
|---------|-----|-------|
| Components | Heavyweight | Lightweight |
| Platform Dependency | Platform-dependent | Platform-independent |
| Look and Feel | Native OS | Pluggable |
| Performance | Faster | Slightly slower |
| Component Set | Limited | Extensive |

## 3. Basic Swing Application Structure

### Simple "Hello World" Swing Application
```java
import javax.swing.*;
import java.awt.*;

public class HelloSwing {
    public static void main(String[] args) {
        // Ensure GUI updates happen on Event Dispatch Thread
        SwingUtilities.invokeLater(() -> {
            createAndShowGUI();
        });
    }
    
    private static void createAndShowGUI() {
        // Create the main frame
        JFrame frame = new JFrame("Hello Swing");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 200);
        frame.setLocationRelativeTo(null); // Center the window
        
        // Create and add a label
        JLabel label = new JLabel("Hello, Swing World!", JLabel.CENTER);
        label.setFont(new Font("Arial", Font.BOLD, 16));
        frame.add(label);
        
        // Make the frame visible
        frame.setVisible(true);
    }
}
```

### Enhanced Swing Application with Multiple Components
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class BasicSwingApp extends JFrame {
    private JTextField textField;
    private JLabel resultLabel;
    private JButton button;
    
    public BasicSwingApp() {
        initializeComponents();
        setupLayout();
        addEventHandlers();
        configureFrame();
    }
    
    private void initializeComponents() {
        textField = new JTextField(20);
        resultLabel = new JLabel("Enter text and click the button");
        button = new JButton("Process Text");
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Create panels for better organization
        JPanel topPanel = new JPanel(new FlowLayout());
        topPanel.add(new JLabel("Input: "));
        topPanel.add(textField);
        topPanel.add(button);
        
        JPanel centerPanel = new JPanel(new FlowLayout());
        centerPanel.add(resultLabel);
        
        add(topPanel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
    }
    
    private void addEventHandlers() {
        button.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String inputText = textField.getText();
                if (!inputText.trim().isEmpty()) {
                    resultLabel.setText("You entered: " + inputText);
                    textField.setText(""); // Clear the input field
                } else {
                    resultLabel.setText("Please enter some text!");
                }
            }
        });
        
        // Alternative using lambda expression (Java 8+)
        // button.addActionListener(e -> processText());
    }
    
    private void configureFrame() {
        setTitle("Basic Swing Application");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 150);
        setLocationRelativeTo(null);
        setResizable(false);
    }
    
    public static void main(String[] args) {
        // Set system look and feel
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeel());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        SwingUtilities.invokeLater(() -> {
            new BasicSwingApp().setVisible(true);
        });
    }
}
```

## 4. Swing Component Hierarchy

### Top-Level Containers
```java
import javax.swing.*;
import java.awt.*;

public class TopLevelContainers {
    
    public static void demonstrateJFrame() {
        JFrame frame = new JFrame("JFrame Example");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 300);
        frame.setLocationRelativeTo(null);
        
        // Add content
        frame.add(new JLabel("This is a JFrame", JLabel.CENTER));
        frame.setVisible(true);
    }
    
    public static void demonstrateJDialog() {
        JFrame parentFrame = new JFrame("Parent Frame");
        parentFrame.setSize(300, 200);
        parentFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        parentFrame.setLocationRelativeTo(null);
        
        JButton button = new JButton("Open Dialog");
        button.addActionListener(e -> {
            JDialog dialog = new JDialog(parentFrame, "Dialog Example", true);
            dialog.setSize(250, 150);
            dialog.setLocationRelativeTo(parentFrame);
            dialog.add(new JLabel("This is a JDialog", JLabel.CENTER));
            dialog.setVisible(true);
        });
        
        parentFrame.add(button);
        parentFrame.setVisible(true);
    }
    
    public static void demonstrateJApplet() {
        // Note: JApplet is deprecated in Java 9+
        // Modern web deployment uses Java Web Start or other technologies
        
        JFrame frame = new JFrame("Applet Container");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 300);
        
        // Simulate applet content
        JPanel appletContent = new JPanel();
        appletContent.setBackground(Color.LIGHT_GRAY);
        appletContent.add(new JLabel("Simulated Applet Content"));
        
        frame.add(appletContent);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            demonstrateJFrame();
            // Uncomment to test other containers
            // demonstrateJDialog();
            // demonstrateJApplet();
        });
    }
}
```

## 5. Event Dispatch Thread (EDT)

### Understanding EDT
The Event Dispatch Thread is responsible for handling all GUI events and updates. All Swing components must be accessed from the EDT for thread safety.

```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class EDTExample extends JFrame {
    private JProgressBar progressBar;
    private JButton startButton;
    private JLabel statusLabel;
    
    public EDTExample() {
        setupComponents();
        setupLayout();
        setupEventHandlers();
        configureFrame();
    }
    
    private void setupComponents() {
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        
        startButton = new JButton("Start Long Task");
        statusLabel = new JLabel("Ready");
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        JPanel topPanel = new JPanel();
        topPanel.add(startButton);
        
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.add(progressBar, BorderLayout.CENTER);
        centerPanel.add(statusLabel, BorderLayout.SOUTH);
        
        add(topPanel, BorderLayout.NORTH);
        add(centerPanel, BorderLayout.CENTER);
    }
    
    private void setupEventHandlers() {
        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                startLongRunningTask();
            }
        });
    }
    
    private void startLongRunningTask() {
        startButton.setEnabled(false);
        statusLabel.setText("Task running...");
        
        // Use SwingWorker for background tasks
        SwingWorker<Void, Integer> worker = new SwingWorker<Void, Integer>() {
            @Override
            protected Void doInBackground() throws Exception {
                for (int i = 0; i <= 100; i++) {
                    // Simulate work
                    Thread.sleep(50);
                    
                    // Publish progress (this will call process() on EDT)
                    publish(i);
                    
                    // Check if task was cancelled
                    if (isCancelled()) {
                        break;
                    }
                }
                return null;
            }
            
            @Override
            protected void process(java.util.List<Integer> chunks) {
                // This runs on EDT - safe to update GUI
                int latestProgress = chunks.get(chunks.size() - 1);
                progressBar.setValue(latestProgress);
            }
            
            @Override
            protected void done() {
                // This runs on EDT when task completes
                startButton.setEnabled(true);
                statusLabel.setText("Task completed!");
                progressBar.setValue(0);
            }
        };
        
        worker.execute();
    }
    
    private void configureFrame() {
        setTitle("EDT Example with SwingWorker");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 150);
        setLocationRelativeTo(null);
    }
    
    // Utility method to check if we're on EDT
    public static void checkEDT(String operation) {
        if (SwingUtilities.isEventDispatchThread()) {
            System.out.println(operation + " - Running on EDT ✓");
        } else {
            System.out.println(operation + " - NOT on EDT ✗");
        }
    }
    
    public static void main(String[] args) {
        // Wrong way - creating GUI on main thread
        // new EDTExample().setVisible(true);
        
        // Correct way - use EDT for GUI operations
        SwingUtilities.invokeLater(() -> {
            checkEDT("GUI Creation");
            new EDTExample().setVisible(true);
        });
        
        // Also correct - using invokeAndWait for synchronous execution
        /*
        try {
            SwingUtilities.invokeAndWait(() -> {
                new EDTExample().setVisible(true);
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
        */
    }
}
```

## 6. Look and Feel

### Setting Different Look and Feel
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class LookAndFeelDemo extends JFrame {
    private JComboBox<String> lafComboBox;
    private JButton applyButton;
    private JPanel contentPanel;
    
    public LookAndFeelDemo() {
        setupComponents();
        setupLayout();
        setupEventHandlers();
        configureFrame();
    }
    
    private void setupComponents() {
        // Get available Look and Feel options
        UIManager.LookAndFeelInfo[] lafs = UIManager.getInstalledLookAndFeels();
        String[] lafNames = new String[lafs.length];
        for (int i = 0; i < lafs.length; i++) {
            lafNames[i] = lafs[i].getName();
        }
        
        lafComboBox = new JComboBox<>(lafNames);
        applyButton = new JButton("Apply Look and Feel");
        
        // Create sample components to demonstrate LAF changes
        contentPanel = new JPanel();
        contentPanel.setLayout(new GridLayout(4, 2, 10, 10));
        contentPanel.setBorder(BorderFactory.createTitledBorder("Sample Components"));
        
        contentPanel.add(new JLabel("Label:"));
        contentPanel.add(new JLabel("Sample Text"));
        
        contentPanel.add(new JLabel("Button:"));
        contentPanel.add(new JButton("Sample Button"));
        
        contentPanel.add(new JLabel("Text Field:"));
        contentPanel.add(new JTextField("Sample Text"));
        
        contentPanel.add(new JLabel("Check Box:"));
        contentPanel.add(new JCheckBox("Sample Checkbox", true));
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        JPanel controlPanel = new JPanel(new FlowLayout());
        controlPanel.add(new JLabel("Look and Feel:"));
        controlPanel.add(lafComboBox);
        controlPanel.add(applyButton);
        
        add(controlPanel, BorderLayout.NORTH);
        add(contentPanel, BorderLayout.CENTER);
    }
    
    private void setupEventHandlers() {
        applyButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                changeLookAndFeel();
            }
        });
    }
    
    private void changeLookAndFeel() {
        String selectedLAF = (String) lafComboBox.getSelectedItem();
        
        try {
            UIManager.LookAndFeelInfo[] lafs = UIManager.getInstalledLookAndFeels();
            for (UIManager.LookAndFeelInfo laf : lafs) {
                if (laf.getName().equals(selectedLAF)) {
                    UIManager.setLookAndFeel(laf.getClassName());
                    SwingUtilities.updateComponentTreeUI(this);
                    break;
                }
            }
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, 
                "Error setting Look and Feel: " + ex.getMessage(),
                "Error", 
                JOptionPane.ERROR_MESSAGE);
        }
    }
    
    private void configureFrame() {
        setTitle("Look and Feel Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 300);
        setLocationRelativeTo(null);
    }
    
    // Static method to set system LAF
    public static void setSystemLookAndFeel() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeel());
        } catch (Exception e) {
            System.err.println("Failed to set system Look and Feel: " + e.getMessage());
        }
    }
    
    // Static method to set cross-platform LAF (Metal)
    public static void setCrossPlatformLookAndFeel() {
        try {
            UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeel());
        } catch (Exception e) {
            System.err.println("Failed to set cross-platform Look and Feel: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        // Optionally set LAF before creating GUI
        setSystemLookAndFeel();
        
        SwingUtilities.invokeLater(() -> {
            new LookAndFeelDemo().setVisible(true);
        });
    }
}
```

## 7. Basic Swing Best Practices

### 1. Thread Safety
- Always create and modify GUI components on the EDT
- Use SwingWorker for long-running tasks
- Use SwingUtilities.invokeLater() or invokeAndWait()

### 2. Memory Management
- Properly dispose of frames and dialogs
- Remove event listeners when no longer needed
- Use weak references for listeners when appropriate

### 3. User Experience
- Provide immediate feedback for user actions
- Use progress indicators for long operations
- Implement proper keyboard navigation and shortcuts

### 4. Code Organization
- Separate GUI creation from business logic
- Use MVC or MVP patterns for complex applications
- Create reusable custom components

## Summary

Swing provides a comprehensive framework for building desktop GUI applications in Java:

### Key Concepts Covered
- **Swing Architecture**: Lightweight components and MVC pattern
- **Component Hierarchy**: Top-level containers and their relationships
- **Event Dispatch Thread**: Thread safety in GUI programming
- **Look and Feel**: Customizable appearance and behavior

### Essential Best Practices
- Use EDT for all GUI operations
- Implement proper event handling
- Follow consistent layout principles
- Provide good user experience with feedback and progress indicators

### Next Steps
In the following sections, we'll dive deeper into:
- Layout Managers for component positioning
- Event Handling mechanisms
- Individual Swing components
- Advanced GUI programming techniques

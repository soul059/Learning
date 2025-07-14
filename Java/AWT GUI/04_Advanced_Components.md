# Advanced AWT Components

## 1. Checkbox Component

Checkbox provides binary selection options (checked/unchecked) and can be grouped for exclusive selection.

### Basic Checkbox Example
```java
import java.awt.*;
import java.awt.event.*;

public class CheckboxDemo extends Frame implements ItemListener {
    private Checkbox check1, check2, check3;
    private CheckboxGroup radioGroup;
    private Checkbox radio1, radio2, radio3;
    private Label statusLabel;
    private TextArea resultArea;
    
    public CheckboxDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        // Individual checkboxes
        check1 = new Checkbox("Java Programming");
        check2 = new Checkbox("Web Development");
        check3 = new Checkbox("Database Design", true); // Initially checked
        
        check1.addItemListener(this);
        check2.addItemListener(this);
        check3.addItemListener(this);
        
        // Radio button group (exclusive selection)
        radioGroup = new CheckboxGroup();
        radio1 = new Checkbox("Beginner", radioGroup, false);
        radio2 = new Checkbox("Intermediate", radioGroup, true); // Initially selected
        radio3 = new Checkbox("Advanced", radioGroup, false);
        
        radio1.addItemListener(this);
        radio2.addItemListener(this);
        radio3.addItemListener(this);
        
        statusLabel = new Label("Make your selections above");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        resultArea = new TextArea(8, 40);
        resultArea.setEditable(false);
        resultArea.setText("Selection summary will appear here...\n");
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        Panel mainPanel = new Panel(new GridLayout(1, 2, 10, 0));
        
        // Left panel - Skills selection
        Panel skillsPanel = new Panel();
        skillsPanel.setLayout(new GridLayout(5, 1, 0, 5));
        skillsPanel.add(new Label("Skills (multiple selection):"));
        skillsPanel.add(check1);
        skillsPanel.add(check2);
        skillsPanel.add(check3);
        
        // Right panel - Experience level
        Panel levelPanel = new Panel();
        levelPanel.setLayout(new GridLayout(5, 1, 0, 5));
        levelPanel.add(new Label("Experience Level (single selection):"));
        levelPanel.add(radio1);
        levelPanel.add(radio2);
        levelPanel.add(radio3);
        
        mainPanel.add(skillsPanel);
        mainPanel.add(levelPanel);
        
        add(mainPanel, BorderLayout.NORTH);
        add(statusLabel, BorderLayout.CENTER);
        add(resultArea, BorderLayout.SOUTH);
        
        updateResult();
    }
    
    @Override
    public void itemStateChanged(ItemEvent e) {
        Checkbox source = (Checkbox) e.getSource();
        String action = (e.getStateChange() == ItemEvent.SELECTED) ? "selected" : "deselected";
        statusLabel.setText(source.getLabel() + " " + action);
        updateResult();
    }
    
    private void updateResult() {
        StringBuilder result = new StringBuilder();
        result.append("=== CURRENT SELECTIONS ===\n\n");
        
        result.append("Skills selected:\n");
        if (check1.getState()) result.append("✓ Java Programming\n");
        if (check2.getState()) result.append("✓ Web Development\n");
        if (check3.getState()) result.append("✓ Database Design\n");
        
        if (!check1.getState() && !check2.getState() && !check3.getState()) {
            result.append("  (No skills selected)\n");
        }
        
        result.append("\nExperience Level:\n");
        Checkbox selectedRadio = radioGroup.getSelectedCheckbox();
        if (selectedRadio != null) {
            result.append("• ").append(selectedRadio.getLabel()).append("\n");
        }
        
        result.append("\n--- Summary ---\n");
        int skillCount = 0;
        if (check1.getState()) skillCount++;
        if (check2.getState()) skillCount++;
        if (check3.getState()) skillCount++;
        
        result.append("Total skills: ").append(skillCount).append("\n");
        result.append("Experience: ").append(selectedRadio != null ? selectedRadio.getLabel() : "None").append("\n");
        
        resultArea.setText(result.toString());
    }
    
    private void configureFrame() {
        setTitle("Checkbox Demo - Skills Survey");
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
        new CheckboxDemo();
    }
}
```

### Interactive Settings Panel
```java
import java.awt.*;
import java.awt.event.*;

public class SettingsPanel extends Frame implements ItemListener, ActionListener {
    // Feature toggles
    private Checkbox enableSound, enableNotifications, enableAutoSave, enableDarkMode;
    
    // Quality settings (radio buttons)
    private CheckboxGroup qualityGroup;
    private Checkbox lowQuality, mediumQuality, highQuality;
    
    // Language settings
    private CheckboxGroup languageGroup;
    private Checkbox englishLang, spanishLang, frenchLang, germanLang;
    
    private Button applyButton, resetButton;
    private Label statusLabel;
    private Panel previewPanel;
    
    public SettingsPanel() {
        setupComponents();
        setupLayout();
        applySettings();
        configureFrame();
    }
    
    private void setupComponents() {
        // Feature checkboxes
        enableSound = new Checkbox("Enable Sound Effects", true);
        enableNotifications = new Checkbox("Enable Notifications", true);
        enableAutoSave = new Checkbox("Auto-save Documents", false);
        enableDarkMode = new Checkbox("Dark Mode", false);
        
        enableSound.addItemListener(this);
        enableNotifications.addItemListener(this);
        enableAutoSave.addItemListener(this);
        enableDarkMode.addItemListener(this);
        
        // Quality radio buttons
        qualityGroup = new CheckboxGroup();
        lowQuality = new Checkbox("Low", qualityGroup, false);
        mediumQuality = new Checkbox("Medium", qualityGroup, true);
        highQuality = new Checkbox("High", qualityGroup, false);
        
        lowQuality.addItemListener(this);
        mediumQuality.addItemListener(this);
        highQuality.addItemListener(this);
        
        // Language radio buttons
        languageGroup = new CheckboxGroup();
        englishLang = new Checkbox("English", languageGroup, true);
        spanishLang = new Checkbox("Español", languageGroup, false);
        frenchLang = new Checkbox("Français", languageGroup, false);
        germanLang = new Checkbox("Deutsch", languageGroup, false);
        
        englishLang.addItemListener(this);
        spanishLang.addItemListener(this);
        frenchLang.addItemListener(this);
        germanLang.addItemListener(this);
        
        // Control buttons
        applyButton = new Button("Apply Settings");
        resetButton = new Button("Reset to Defaults");
        
        applyButton.addActionListener(this);
        resetButton.addActionListener(this);
        
        statusLabel = new Label("Configure your application settings");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        // Preview panel
        previewPanel = new Panel();
        previewPanel.setBackground(Color.WHITE);
        previewPanel.add(new Label("Preview Area"));
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        Panel mainPanel = new Panel(new GridLayout(1, 3, 10, 0));
        
        // Features panel
        Panel featuresPanel = new Panel();
        featuresPanel.setLayout(new GridLayout(6, 1, 0, 5));
        featuresPanel.add(new Label("Features:"));
        featuresPanel.add(enableSound);
        featuresPanel.add(enableNotifications);
        featuresPanel.add(enableAutoSave);
        featuresPanel.add(enableDarkMode);
        
        // Quality panel
        Panel qualityPanel = new Panel();
        qualityPanel.setLayout(new GridLayout(5, 1, 0, 5));
        qualityPanel.add(new Label("Graphics Quality:"));
        qualityPanel.add(lowQuality);
        qualityPanel.add(mediumQuality);
        qualityPanel.add(highQuality);
        
        // Language panel
        Panel languagePanel = new Panel();
        languagePanel.setLayout(new GridLayout(6, 1, 0, 5));
        languagePanel.add(new Label("Language:"));
        languagePanel.add(englishLang);
        languagePanel.add(spanishLang);
        languagePanel.add(frenchLang);
        languagePanel.add(germanLang);
        
        mainPanel.add(featuresPanel);
        mainPanel.add(qualityPanel);
        mainPanel.add(languagePanel);
        
        // Button panel
        Panel buttonPanel = new Panel(new FlowLayout());
        buttonPanel.add(applyButton);
        buttonPanel.add(resetButton);
        
        // Preview area
        previewPanel.setPreferredSize(new Dimension(100, 60));
        
        add(mainPanel, BorderLayout.NORTH);
        add(buttonPanel, BorderLayout.CENTER);
        add(previewPanel, BorderLayout.EAST);
        add(statusLabel, BorderLayout.SOUTH);
    }
    
    @Override
    public void itemStateChanged(ItemEvent e) {
        Checkbox source = (Checkbox) e.getSource();
        String status = (e.getStateChange() == ItemEvent.SELECTED) ? "enabled" : "disabled";
        statusLabel.setText(source.getLabel() + " " + status);
        
        // Real-time preview updates
        applySettings();
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == applyButton) {
            applySettings();
            statusLabel.setText("Settings applied successfully!");
        } else if (e.getSource() == resetButton) {
            resetToDefaults();
            statusLabel.setText("Settings reset to defaults");
        }
    }
    
    private void applySettings() {
        // Update preview panel based on settings
        if (enableDarkMode.getState()) {
            previewPanel.setBackground(Color.DARK_GRAY);
            previewPanel.setForeground(Color.WHITE);
        } else {
            previewPanel.setBackground(Color.WHITE);
            previewPanel.setForeground(Color.BLACK);
        }
        
        // Update preview content
        previewPanel.removeAll();
        Label previewLabel = new Label("Preview");
        previewLabel.setForeground(previewPanel.getForeground());
        
        Checkbox selectedLang = languageGroup.getSelectedCheckbox();
        if (selectedLang != null) {
            switch (selectedLang.getLabel()) {
                case "Español":
                    previewLabel.setText("Vista Previa");
                    break;
                case "Français":
                    previewLabel.setText("Aperçu");
                    break;
                case "Deutsch":
                    previewLabel.setText("Vorschau");
                    break;
                default:
                    previewLabel.setText("Preview");
                    break;
            }
        }
        
        previewPanel.add(previewLabel);
        previewPanel.revalidate();
        
        // Simulate other settings effects
        printCurrentSettings();
    }
    
    private void resetToDefaults() {
        enableSound.setState(true);
        enableNotifications.setState(true);
        enableAutoSave.setState(false);
        enableDarkMode.setState(false);
        
        qualityGroup.setSelectedCheckbox(mediumQuality);
        languageGroup.setSelectedCheckbox(englishLang);
        
        applySettings();
    }
    
    private void printCurrentSettings() {
        System.out.println("=== Current Settings ===");
        System.out.println("Sound: " + enableSound.getState());
        System.out.println("Notifications: " + enableNotifications.getState());
        System.out.println("Auto-save: " + enableAutoSave.getState());
        System.out.println("Dark Mode: " + enableDarkMode.getState());
        
        Checkbox quality = qualityGroup.getSelectedCheckbox();
        System.out.println("Quality: " + (quality != null ? quality.getLabel() : "None"));
        
        Checkbox language = languageGroup.getSelectedCheckbox();
        System.out.println("Language: " + (language != null ? language.getLabel() : "None"));
    }
    
    private void configureFrame() {
        setTitle("Application Settings - Checkbox Controls");
        setSize(600, 350);
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
        new SettingsPanel();
    }
}
```

## 2. Choice Component (Dropdown)

Choice component provides a dropdown list for selecting one item from multiple options.

### Choice Component Demo
```java
import java.awt.*;
import java.awt.event.*;

public class ChoiceDemo extends Frame implements ItemListener, ActionListener {
    private Choice countryChoice, cityChoice, languageChoice;
    private Button addButton, removeButton;
    private TextField newItemField;
    private Label statusLabel;
    private TextArea infoArea;
    
    public ChoiceDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        // Country choice
        countryChoice = new Choice();
        countryChoice.add("Select Country");
        countryChoice.add("United States");
        countryChoice.add("Canada");
        countryChoice.add("United Kingdom");
        countryChoice.add("Germany");
        countryChoice.add("France");
        countryChoice.add("Japan");
        countryChoice.add("Australia");
        countryChoice.addItemListener(this);
        
        // City choice (initially empty)
        cityChoice = new Choice();
        cityChoice.add("Select country first");
        cityChoice.setEnabled(false);
        cityChoice.addItemListener(this);
        
        // Programming language choice
        languageChoice = new Choice();
        languageChoice.add("Java");
        languageChoice.add("Python");
        languageChoice.add("JavaScript");
        languageChoice.add("C++");
        languageChoice.add("C#");
        languageChoice.select("Java"); // Pre-select Java
        languageChoice.addItemListener(this);
        
        // Control components
        newItemField = new TextField("Enter new language", 15);
        addButton = new Button("Add Language");
        removeButton = new Button("Remove Selected");
        
        addButton.addActionListener(this);
        removeButton.addActionListener(this);
        
        statusLabel = new Label("Make selections from the dropdowns above");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        infoArea = new TextArea(10, 50);
        infoArea.setEditable(false);
        infoArea.setText("Selection Information:\n" +
                        "====================\n" +
                        "Make selections to see details here...\n");
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Top panel with choices
        Panel choicePanel = new Panel(new GridLayout(3, 2, 5, 5));
        choicePanel.add(new Label("Country:"));
        choicePanel.add(countryChoice);
        choicePanel.add(new Label("City:"));
        choicePanel.add(cityChoice);
        choicePanel.add(new Label("Programming Language:"));
        choicePanel.add(languageChoice);
        
        // Control panel for adding/removing languages
        Panel controlPanel = new Panel(new FlowLayout());
        controlPanel.add(new Label("Manage Languages:"));
        controlPanel.add(newItemField);
        controlPanel.add(addButton);
        controlPanel.add(removeButton);
        
        // Main layout
        add(choicePanel, BorderLayout.NORTH);
        add(controlPanel, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
        add(infoArea, BorderLayout.EAST);
        
        updateInfo();
    }
    
    @Override
    public void itemStateChanged(ItemEvent e) {
        if (e.getSource() == countryChoice) {
            updateCities();
        }
        
        updateInfo();
        
        Choice source = (Choice) e.getSource();
        statusLabel.setText("Selected: " + source.getSelectedItem());
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == addButton) {
            addLanguage();
        } else if (e.getSource() == removeButton) {
            removeLanguage();
        }
    }
    
    private void updateCities() {
        String selectedCountry = countryChoice.getSelectedItem();
        
        cityChoice.removeAll();
        cityChoice.setEnabled(false);
        
        if (selectedCountry.equals("Select Country")) {
            cityChoice.add("Select country first");
            return;
        }
        
        // Add cities based on selected country
        switch (selectedCountry) {
            case "United States":
                cityChoice.add("New York");
                cityChoice.add("Los Angeles");
                cityChoice.add("Chicago");
                cityChoice.add("Houston");
                cityChoice.add("Phoenix");
                break;
            case "Canada":
                cityChoice.add("Toronto");
                cityChoice.add("Vancouver");
                cityChoice.add("Montreal");
                cityChoice.add("Calgary");
                cityChoice.add("Ottawa");
                break;
            case "United Kingdom":
                cityChoice.add("London");
                cityChoice.add("Manchester");
                cityChoice.add("Birmingham");
                cityChoice.add("Liverpool");
                cityChoice.add("Leeds");
                break;
            case "Germany":
                cityChoice.add("Berlin");
                cityChoice.add("Munich");
                cityChoice.add("Hamburg");
                cityChoice.add("Cologne");
                cityChoice.add("Frankfurt");
                break;
            case "France":
                cityChoice.add("Paris");
                cityChoice.add("Lyon");
                cityChoice.add("Marseille");
                cityChoice.add("Toulouse");
                cityChoice.add("Nice");
                break;
            case "Japan":
                cityChoice.add("Tokyo");
                cityChoice.add("Osaka");
                cityChoice.add("Kyoto");
                cityChoice.add("Yokohama");
                cityChoice.add("Kobe");
                break;
            case "Australia":
                cityChoice.add("Sydney");
                cityChoice.add("Melbourne");
                cityChoice.add("Brisbane");
                cityChoice.add("Perth");
                cityChoice.add("Adelaide");
                break;
            default:
                cityChoice.add("No cities available");
                break;
        }
        
        cityChoice.setEnabled(true);
        cityChoice.select(0); // Select first city
    }
    
    private void addLanguage() {
        String newLanguage = newItemField.getText().trim();
        
        if (newLanguage.isEmpty()) {
            statusLabel.setText("Please enter a language name");
            return;
        }
        
        // Check if language already exists
        for (int i = 0; i < languageChoice.getItemCount(); i++) {
            if (languageChoice.getItem(i).equalsIgnoreCase(newLanguage)) {
                statusLabel.setText("Language '" + newLanguage + "' already exists");
                return;
            }
        }
        
        languageChoice.add(newLanguage);
        languageChoice.select(newLanguage);
        newItemField.setText("");
        statusLabel.setText("Added language: " + newLanguage);
        updateInfo();
    }
    
    private void removeLanguage() {
        String selectedLanguage = languageChoice.getSelectedItem();
        
        if (languageChoice.getItemCount() <= 1) {
            statusLabel.setText("Cannot remove - at least one language must remain");
            return;
        }
        
        // Find and remove the selected item
        for (int i = 0; i < languageChoice.getItemCount(); i++) {
            if (languageChoice.getItem(i).equals(selectedLanguage)) {
                languageChoice.remove(i);
                break;
            }
        }
        
        // Select first remaining item
        if (languageChoice.getItemCount() > 0) {
            languageChoice.select(0);
        }
        
        statusLabel.setText("Removed language: " + selectedLanguage);
        updateInfo();
    }
    
    private void updateInfo() {
        StringBuilder info = new StringBuilder();
        info.append("Current Selections:\n");
        info.append("==================\n\n");
        
        info.append("Country: ").append(countryChoice.getSelectedItem()).append("\n");
        info.append("City: ").append(cityChoice.getSelectedItem()).append("\n");
        info.append("Language: ").append(languageChoice.getSelectedItem()).append("\n\n");
        
        info.append("Available Options:\n");
        info.append("------------------\n");
        
        info.append("Countries (").append(countryChoice.getItemCount()).append("):\n");
        for (int i = 0; i < countryChoice.getItemCount(); i++) {
            info.append("  • ").append(countryChoice.getItem(i)).append("\n");
        }
        
        info.append("\nCities (").append(cityChoice.getItemCount()).append("):\n");
        for (int i = 0; i < cityChoice.getItemCount(); i++) {
            info.append("  • ").append(cityChoice.getItem(i)).append("\n");
        }
        
        info.append("\nLanguages (").append(languageChoice.getItemCount()).append("):\n");
        for (int i = 0; i < languageChoice.getItemCount(); i++) {
            String lang = languageChoice.getItem(i);
            if (lang.equals(languageChoice.getSelectedItem())) {
                info.append("  ▶ ").append(lang).append(" (selected)\n");
            } else {
                info.append("  • ").append(lang).append("\n");
            }
        }
        
        infoArea.setText(info.toString());
    }
    
    private void configureFrame() {
        setTitle("Choice Component Demo - Location & Language Selector");
        setSize(700, 400);
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
        new ChoiceDemo();
    }
}
```

## 3. List Component

List component displays multiple items and allows single or multiple selections.

### List Component Demo
```java
import java.awt.*;
import java.awt.event.*;

public class ListDemo extends Frame implements ActionListener, ItemListener {
    private List singleList, multiList;
    private Button addButton, removeButton, clearButton, moveButton;
    private TextField itemField;
    private Label statusLabel;
    private TextArea detailArea;
    
    public ListDemo() {
        setupComponents();
        setupLayout();
        configureFrame();
    }
    
    private void setupComponents() {
        // Single selection list
        singleList = new List(8, false); // 8 visible rows, single selection
        singleList.add("Apple");
        singleList.add("Banana");
        singleList.add("Cherry");
        singleList.add("Date");
        singleList.add("Elderberry");
        singleList.add("Fig");
        singleList.add("Grape");
        singleList.addItemListener(this);
        
        // Multiple selection list
        multiList = new List(8, true); // 8 visible rows, multiple selection
        multiList.add("Red");
        multiList.add("Green");
        multiList.add("Blue");
        multiList.add("Yellow");
        multiList.add("Orange");
        multiList.add("Purple");
        multiList.add("Pink");
        multiList.add("Cyan");
        multiList.addItemListener(this);
        
        // Control components
        itemField = new TextField("Enter new item", 20);
        addButton = new Button("Add Item");
        removeButton = new Button("Remove Selected");
        clearButton = new Button("Clear All");
        moveButton = new Button("Move Selected →");
        
        addButton.addActionListener(this);
        removeButton.addActionListener(this);
        clearButton.addActionListener(this);
        moveButton.addActionListener(this);
        
        statusLabel = new Label("Select items from the lists");
        statusLabel.setBackground(Color.LIGHT_GRAY);
        
        detailArea = new TextArea(6, 40);
        detailArea.setEditable(false);
        updateDetails();
    }
    
    private void setupLayout() {
        setLayout(new BorderLayout());
        
        // Lists panel
        Panel listsPanel = new Panel(new GridLayout(1, 3, 10, 0));
        
        // Single selection panel
        Panel singlePanel = new Panel(new BorderLayout());
        singlePanel.add(new Label("Fruits (Single Selection)", Label.CENTER), BorderLayout.NORTH);
        singlePanel.add(singleList, BorderLayout.CENTER);
        
        // Move button panel
        Panel movePanel = new Panel(new FlowLayout());
        movePanel.add(moveButton);
        
        // Multiple selection panel
        Panel multiPanel = new Panel(new BorderLayout());
        multiPanel.add(new Label("Colors (Multiple Selection)", Label.CENTER), BorderLayout.NORTH);
        multiPanel.add(multiList, BorderLayout.CENTER);
        
        listsPanel.add(singlePanel);
        listsPanel.add(movePanel);
        listsPanel.add(multiPanel);
        
        // Control panel
        Panel controlPanel = new Panel(new FlowLayout());
        controlPanel.add(new Label("Add Item:"));
        controlPanel.add(itemField);
        controlPanel.add(addButton);
        controlPanel.add(removeButton);
        controlPanel.add(clearButton);
        
        add(listsPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.NORTH);
        add(statusLabel, BorderLayout.SOUTH);
        add(detailArea, BorderLayout.EAST);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == addButton) {
            addItem();
        } else if (e.getSource() == removeButton) {
            removeSelectedItems();
        } else if (e.getSource() == clearButton) {
            clearLists();
        } else if (e.getSource() == moveButton) {
            moveItems();
        }
        updateDetails();
    }
    
    @Override
    public void itemStateChanged(ItemEvent e) {
        List source = (List) e.getSource();
        String listName = (source == singleList) ? "Fruits" : "Colors";
        
        if (e.getStateChange() == ItemEvent.SELECTED) {
            statusLabel.setText("Selected in " + listName + ": " + (String) e.getItem());
        } else {
            statusLabel.setText("Deselected in " + listName + ": " + (String) e.getItem());
        }
        
        updateDetails();
    }
    
    private void addItem() {
        String newItem = itemField.getText().trim();
        
        if (newItem.isEmpty()) {
            statusLabel.setText("Please enter an item to add");
            return;
        }
        
        // Add to the list that has focus or the single list by default
        singleList.add(newItem);
        itemField.setText("");
        statusLabel.setText("Added item: " + newItem);
    }
    
    private void removeSelectedItems() {
        // Remove from single list
        int singleIndex = singleList.getSelectedIndex();
        if (singleIndex != -1) {
            String removedItem = singleList.getItem(singleIndex);
            singleList.remove(singleIndex);
            statusLabel.setText("Removed from fruits: " + removedItem);
        }
        
        // Remove from multi list (remove all selected)
        int[] multiIndices = multiList.getSelectedIndexes();
        if (multiIndices.length > 0) {
            // Remove in reverse order to maintain indices
            for (int i = multiIndices.length - 1; i >= 0; i--) {
                multiList.remove(multiIndices[i]);
            }
            statusLabel.setText("Removed " + multiIndices.length + " items from colors");
        }
        
        if (singleIndex == -1 && multiIndices.length == 0) {
            statusLabel.setText("No items selected for removal");
        }
    }
    
    private void clearLists() {
        singleList.removeAll();
        multiList.removeAll();
        statusLabel.setText("All lists cleared");
    }
    
    private void moveItems() {
        // Move selected item from single list to multi list
        int selectedIndex = singleList.getSelectedIndex();
        if (selectedIndex != -1) {
            String item = singleList.getItem(selectedIndex);
            singleList.remove(selectedIndex);
            multiList.add(item);
            statusLabel.setText("Moved '" + item + "' to colors list");
        } else {
            statusLabel.setText("Select an item from fruits list to move");
        }
    }
    
    private void updateDetails() {
        StringBuilder details = new StringBuilder();
        details.append("LIST DETAILS\n");
        details.append("============\n\n");
        
        // Single list details
        details.append("FRUITS LIST:\n");
        details.append("Items: ").append(singleList.getItemCount()).append("\n");
        details.append("Selected: ");
        int singleSelected = singleList.getSelectedIndex();
        if (singleSelected != -1) {
            details.append(singleList.getItem(singleSelected));
        } else {
            details.append("None");
        }
        details.append("\n\nAll fruits:\n");
        for (int i = 0; i < singleList.getItemCount(); i++) {
            details.append("  ").append(i + 1).append(". ").append(singleList.getItem(i));
            if (i == singleSelected) {
                details.append(" ★");
            }
            details.append("\n");
        }
        
        // Multi list details
        details.append("\nCOLORS LIST:\n");
        details.append("Items: ").append(multiList.getItemCount()).append("\n");
        int[] multiSelected = multiList.getSelectedIndexes();
        details.append("Selected: ").append(multiSelected.length).append("\n\n");
        details.append("All colors:\n");
        for (int i = 0; i < multiList.getItemCount(); i++) {
            details.append("  ").append(i + 1).append(". ").append(multiList.getItem(i));
            
            // Check if this item is selected
            for (int selectedIndex : multiSelected) {
                if (i == selectedIndex) {
                    details.append(" ★");
                    break;
                }
            }
            details.append("\n");
        }
        
        if (multiSelected.length > 0) {
            details.append("\nSelected colors:\n");
            for (int index : multiSelected) {
                details.append("  • ").append(multiList.getItem(index)).append("\n");
            }
        }
        
        detailArea.setText(details.toString());
    }
    
    private void configureFrame() {
        setTitle("List Component Demo - Fruits and Colors");
        setSize(800, 450);
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
        new ListDemo();
    }
}
```

## Summary

Advanced AWT Components provide sophisticated user interaction capabilities:

### Component Comparison

| Component | Selection Type | Best Use Case | Key Features |
|-----------|---------------|---------------|--------------|
| **Checkbox** | Binary/Multiple | Settings, features | Individual toggles, radio groups |
| **Choice** | Single from dropdown | Space-efficient selection | Dropdown list, dynamic content |
| **List** | Single/Multiple visible | Large datasets | Scrollable, multiple selection modes |

### Event Handling Patterns

1. **ItemListener**: Handles selection changes in checkboxes, choices, and lists
2. **ActionListener**: Handles button clicks and control actions
3. **Real-time Updates**: Immediate feedback as selections change
4. **Validation**: Check for valid selections before processing

### Best Practices

1. **User Feedback**: Provide clear indication of current selections
2. **Validation**: Check for required selections and handle edge cases
3. **Dynamic Content**: Update dependent components based on selections
4. **Accessibility**: Ensure components are keyboard navigable
5. **Performance**: Use appropriate component for data size and interaction needs

### Common Implementation Patterns

- **Master-Detail**: Selection in one component updates another
- **Settings Panels**: Checkboxes for feature toggles with preview
- **Data Entry**: Lists and choices for predefined options
- **Filtering**: Dynamic content based on selections

These advanced components enable creation of sophisticated user interfaces with rich interaction capabilities while maintaining the native look and feel of the operating system.

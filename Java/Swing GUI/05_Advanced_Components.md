# Advanced Swing Components

## 1. JComboBox - Dropdown Selection

JComboBox provides a dropdown list for selecting from multiple options.

### Basic JComboBox Examples
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JComboBoxDemo extends JFrame {
    private JComboBox<String> countryCombo;
    private JComboBox<String> cityCombo;
    private JComboBox<Font> fontCombo;
    private JLabel resultLabel;
    private JTextArea previewArea;
    
    public JComboBoxDemo() {
        setupComboBoxes();
        configureFrame();
    }
    
    private void setupComboBoxes() {
        setLayout(new BorderLayout());
        
        JPanel topPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        
        // Country selection with dependent city combo
        gbc.gridx = 0; gbc.gridy = 0;
        topPanel.add(new JLabel("Country:"), gbc);
        
        gbc.gridx = 1;
        String[] countries = {"Select Country", "USA", "Canada", "UK", "Germany", "France", "Japan"};
        countryCombo = new JComboBox<>(countries);
        countryCombo.addActionListener(e -> updateCities());
        topPanel.add(countryCombo, gbc);
        
        gbc.gridx = 0; gbc.gridy = 1;
        topPanel.add(new JLabel("City:"), gbc);
        
        gbc.gridx = 1;
        cityCombo = new JComboBox<>();
        cityCombo.setEnabled(false);
        cityCombo.addActionListener(e -> updateResult());
        topPanel.add(cityCombo, gbc);
        
        // Font selection combo
        gbc.gridx = 0; gbc.gridy = 2;
        topPanel.add(new JLabel("Font:"), gbc);
        
        gbc.gridx = 1;
        GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
        Font[] fonts = ge.getAllFonts();
        Font[] commonFonts = {
            new Font("Arial", Font.PLAIN, 12),
            new Font("Courier New", Font.PLAIN, 12),
            new Font("Times New Roman", Font.PLAIN, 12),
            new Font("Verdana", Font.PLAIN, 12),
            new Font("Comic Sans MS", Font.PLAIN, 12)
        };
        
        fontCombo = new JComboBox<>(commonFonts);
        fontCombo.setRenderer(new FontComboRenderer());
        fontCombo.addActionListener(e -> updatePreview());
        topPanel.add(fontCombo, gbc);
        
        // Result label
        resultLabel = new JLabel("Make your selections above");
        resultLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Selection Result"),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        
        // Preview area
        previewArea = new JTextArea("This is a font preview area.\nType here to see the selected font in action.");
        previewArea.setRows(8);
        previewArea.setFont(new Font("Arial", Font.PLAIN, 12));
        JScrollPane scrollPane = new JScrollPane(previewArea);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Font Preview"));
        
        add(topPanel, BorderLayout.NORTH);
        add(resultLabel, BorderLayout.CENTER);
        add(scrollPane, BorderLayout.SOUTH);
    }
    
    private void updateCities() {
        String selectedCountry = (String) countryCombo.getSelectedItem();
        cityCombo.removeAllItems();
        
        if (selectedCountry == null || selectedCountry.equals("Select Country")) {
            cityCombo.setEnabled(false);
            updateResult();
            return;
        }
        
        List<String> cities = getCitiesForCountry(selectedCountry);
        for (String city : cities) {
            cityCombo.addItem(city);
        }
        
        cityCombo.setEnabled(true);
        updateResult();
    }
    
    private List<String> getCitiesForCountry(String country) {
        switch (country) {
            case "USA":
                return Arrays.asList("New York", "Los Angeles", "Chicago", "Houston", "Phoenix");
            case "Canada":
                return Arrays.asList("Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa");
            case "UK":
                return Arrays.asList("London", "Manchester", "Birmingham", "Liverpool", "Leeds");
            case "Germany":
                return Arrays.asList("Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt");
            case "France":
                return Arrays.asList("Paris", "Lyon", "Marseille", "Toulouse", "Nice");
            case "Japan":
                return Arrays.asList("Tokyo", "Osaka", "Kyoto", "Yokohama", "Kobe");
            default:
                return new ArrayList<>();
        }
    }
    
    private void updateResult() {
        String country = (String) countryCombo.getSelectedItem();
        String city = (String) cityCombo.getSelectedItem();
        
        if (country == null || country.equals("Select Country")) {
            resultLabel.setText("Please select a country");
        } else if (city == null) {
            resultLabel.setText("Selected: " + country + " (select a city)");
        } else {
            resultLabel.setText("Selected: " + city + ", " + country);
        }
    }
    
    private void updatePreview() {
        Font selectedFont = (Font) fontCombo.getSelectedItem();
        if (selectedFont != null) {
            Font previewFont = selectedFont.deriveFont(14f);
            previewArea.setFont(previewFont);
        }
    }
    
    // Custom renderer for font combo box
    private class FontComboRenderer extends DefaultListCellRenderer {
        @Override
        public Component getListCellRendererComponent(JList<?> list, Object value,
                int index, boolean isSelected, boolean cellHasFocus) {
            
            super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
            
            if (value instanceof Font) {
                Font font = (Font) value;
                setText(font.getName());
                setFont(font.deriveFont(12f));
            }
            
            return this;
        }
    }
    
    private void configureFrame() {
        setTitle("JComboBox Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JComboBoxDemo().setVisible(true);
        });
    }
}
```

### Editable JComboBox with Auto-completion
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class EditableComboBoxDemo extends JFrame {
    private JComboBox<String> editableCombo;
    private JLabel suggestionLabel;
    private List<String> allItems;
    
    public EditableComboBoxDemo() {
        setupEditableCombo();
        configureFrame();
    }
    
    private void setupEditableCombo() {
        setLayout(new BorderLayout());
        
        // Initialize data
        allItems = Arrays.asList(
            "Apple", "Apricot", "Banana", "Blueberry", "Cherry", "Date",
            "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon",
            "Mango", "Orange", "Papaya", "Quince", "Raspberry", "Strawberry"
        );
        
        JPanel inputPanel = new JPanel(new FlowLayout());
        inputPanel.setBorder(BorderFactory.createTitledBorder("Type to search fruits"));
        
        editableCombo = new JComboBox<>(allItems.toArray(new String[0]));
        editableCombo.setEditable(true);
        editableCombo.setPreferredSize(new Dimension(200, 25));
        
        // Add key listener for auto-completion
        JTextField textField = (JTextField) editableCombo.getEditor().getEditorComponent();
        textField.addKeyListener(new KeyAdapter() {
            @Override
            public void keyReleased(KeyEvent e) {
                SwingUtilities.invokeLater(() -> autoComplete());
            }
        });
        
        editableCombo.addActionListener(e -> {
            if (e.getActionCommand().equals("comboBoxEdited") || 
                e.getActionCommand().equals("comboBoxChanged")) {
                String selected = (String) editableCombo.getSelectedItem();
                suggestionLabel.setText("Selected: " + selected);
            }
        });
        
        JButton addButton = new JButton("Add to List");
        addButton.addActionListener(e -> {
            String newItem = (String) editableCombo.getSelectedItem();
            if (newItem != null && !newItem.trim().isEmpty() && !allItems.contains(newItem)) {
                allItems.add(newItem);
                allItems.sort(String::compareToIgnoreCase);
                updateComboBoxModel();
                suggestionLabel.setText("Added: " + newItem);
            }
        });
        
        inputPanel.add(new JLabel("Fruit:"));
        inputPanel.add(editableCombo);
        inputPanel.add(addButton);
        
        suggestionLabel = new JLabel("Start typing to see suggestions...");
        suggestionLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Status"),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        
        // Instructions
        JTextArea instructions = new JTextArea(
            "Instructions:\n" +
            "• Type in the combo box to see auto-completion\n" +
            "• Select from dropdown or type a new fruit name\n" +
            "• Click 'Add to List' to add new items\n" +
            "• Use arrow keys to navigate suggestions"
        );
        instructions.setEditable(false);
        instructions.setBackground(getBackground());
        instructions.setBorder(BorderFactory.createTitledBorder("How to Use"));
        
        add(inputPanel, BorderLayout.NORTH);
        add(suggestionLabel, BorderLayout.CENTER);
        add(instructions, BorderLayout.SOUTH);
    }
    
    private void autoComplete() {
        JTextField textField = (JTextField) editableCombo.getEditor().getEditorComponent();
        String text = textField.getText();
        
        if (text.isEmpty()) {
            updateComboBoxModel();
            return;
        }
        
        // Filter items based on input
        List<String> filteredItems = allItems.stream()
            .filter(item -> item.toLowerCase().startsWith(text.toLowerCase()))
            .collect(Collectors.toList());
        
        if (!filteredItems.isEmpty()) {
            DefaultComboBoxModel<String> model = new DefaultComboBoxModel<>(
                filteredItems.toArray(new String[0])
            );
            editableCombo.setModel(model);
            editableCombo.showPopup();
            
            // Restore text and set selection
            textField.setText(text);
            textField.setCaretPosition(text.length());
            
            suggestionLabel.setText("Found " + filteredItems.size() + " matches");
        } else {
            suggestionLabel.setText("No matches found");
        }
    }
    
    private void updateComboBoxModel() {
        DefaultComboBoxModel<String> model = new DefaultComboBoxModel<>(
            allItems.toArray(new String[0])
        );
        editableCombo.setModel(model);
    }
    
    private void configureFrame() {
        setTitle("Editable JComboBox with Auto-completion");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new EditableComboBoxDemo().setVisible(true);
        });
    }
}
```

## 2. JList - List Selection

JList displays a list of items and allows single or multiple selections.

### JList Examples
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JListDemo extends JFrame {
    private JList<String> singleSelectionList;
    private JList<String> multiSelectionList;
    private JList<ListItem> customList;
    private JLabel selectionLabel;
    
    public JListDemo() {
        setupLists();
        configureFrame();
    }
    
    private void setupLists() {
        setLayout(new BorderLayout());
        
        JPanel mainPanel = new JPanel(new GridLayout(1, 3, 10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Single selection list
        String[] programming_languages = {
            "Java", "Python", "C++", "JavaScript", "C#", "Go", "Rust", "Swift",
            "Kotlin", "TypeScript", "PHP", "Ruby", "Scala", "Dart"
        };
        
        singleSelectionList = new JList<>(programming_languages);
        singleSelectionList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        singleSelectionList.setSelectedIndex(0);
        singleSelectionList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                updateSelectionInfo();
            }
        });
        
        JScrollPane singleScrollPane = new JScrollPane(singleSelectionList);
        singleScrollPane.setBorder(BorderFactory.createTitledBorder("Single Selection"));
        singleScrollPane.setPreferredSize(new Dimension(150, 200));
        
        // Multiple selection list
        String[] colors = {
            "Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink",
            "Cyan", "Magenta", "Gray", "Black", "White", "Brown", "Lime"
        };
        
        multiSelectionList = new JList<>(colors);
        multiSelectionList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
        multiSelectionList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                updateSelectionInfo();
            }
        });
        
        JScrollPane multiScrollPane = new JScrollPane(multiSelectionList);
        multiScrollPane.setBorder(BorderFactory.createTitledBorder("Multiple Selection"));
        multiScrollPane.setPreferredSize(new Dimension(150, 200));
        
        // Custom object list with custom renderer
        List<ListItem> items = Arrays.asList(
            new ListItem("Important Task", "High priority item", true),
            new ListItem("Meeting Notes", "Weekly team meeting", false),
            new ListItem("Code Review", "Review pull request #123", true),
            new ListItem("Documentation", "Update API documentation", false),
            new ListItem("Bug Fix", "Fix login issue", true),
            new ListItem("Feature Request", "Add dark mode support", false)
        );
        
        customList = new JList<>(items.toArray(new ListItem[0]));
        customList.setCellRenderer(new CustomListCellRenderer());
        customList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        customList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                updateSelectionInfo();
            }
        });
        
        // Add double-click handler for custom list
        customList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) {
                    ListItem selected = customList.getSelectedValue();
                    if (selected != null) {
                        selected.togglePriority();
                        customList.repaint();
                        JOptionPane.showMessageDialog(JListDemo.this,
                            "Priority toggled for: " + selected.getTitle());
                    }
                }
            }
        });
        
        JScrollPane customScrollPane = new JScrollPane(customList);
        customScrollPane.setBorder(BorderFactory.createTitledBorder("Custom Items (Double-click to toggle)"));
        customScrollPane.setPreferredSize(new Dimension(200, 200));
        
        mainPanel.add(singleScrollPane);
        mainPanel.add(multiScrollPane);
        mainPanel.add(customScrollPane);
        
        // Selection info panel
        selectionLabel = new JLabel("Make selections in the lists above");
        selectionLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Selection Information"),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        
        // Control buttons
        JPanel buttonPanel = new JPanel(new FlowLayout());
        
        JButton clearButton = new JButton("Clear All Selections");
        clearButton.addActionListener(e -> {
            singleSelectionList.clearSelection();
            multiSelectionList.clearSelection();
            customList.clearSelection();
        });
        
        JButton selectAllButton = new JButton("Select All Colors");
        selectAllButton.addActionListener(e -> {
            multiSelectionList.setSelectionInterval(0, multiSelectionList.getModel().getSize() - 1);
        });
        
        JButton randomButton = new JButton("Random Selection");
        randomButton.addActionListener(e -> {
            int randomIndex = (int) (Math.random() * singleSelectionList.getModel().getSize());
            singleSelectionList.setSelectedIndex(randomIndex);
        });
        
        buttonPanel.add(clearButton);
        buttonPanel.add(selectAllButton);
        buttonPanel.add(randomButton);
        
        add(mainPanel, BorderLayout.CENTER);
        add(selectionLabel, BorderLayout.SOUTH);
        add(buttonPanel, BorderLayout.NORTH);
        
        updateSelectionInfo();
    }
    
    private void updateSelectionInfo() {
        StringBuilder info = new StringBuilder("<html>");
        
        // Single selection info
        String selectedLanguage = singleSelectionList.getSelectedValue();
        info.append("<b>Programming Language:</b> ")
            .append(selectedLanguage != null ? selectedLanguage : "None")
            .append("<br>");
        
        // Multiple selection info
        List<String> selectedColors = multiSelectionList.getSelectedValuesList();
        info.append("<b>Colors:</b> ");
        if (selectedColors.isEmpty()) {
            info.append("None");
        } else {
            info.append(String.join(", ", selectedColors));
        }
        info.append("<br>");
        
        // Custom item info
        ListItem selectedItem = customList.getSelectedValue();
        info.append("<b>Task:</b> ");
        if (selectedItem != null) {
            info.append(selectedItem.getTitle())
                .append(" (Priority: ")
                .append(selectedItem.isHighPriority() ? "High" : "Normal")
                .append(")");
        } else {
            info.append("None");
        }
        
        info.append("</html>");
        selectionLabel.setText(info.toString());
    }
    
    // Custom data class for list items
    private static class ListItem {
        private String title;
        private String description;
        private boolean highPriority;
        
        public ListItem(String title, String description, boolean highPriority) {
            this.title = title;
            this.description = description;
            this.highPriority = highPriority;
        }
        
        public String getTitle() { return title; }
        public String getDescription() { return description; }
        public boolean isHighPriority() { return highPriority; }
        public void togglePriority() { highPriority = !highPriority; }
        
        @Override
        public String toString() {
            return title;
        }
    }
    
    // Custom cell renderer
    private class CustomListCellRenderer extends DefaultListCellRenderer {
        @Override
        public Component getListCellRendererComponent(JList<?> list, Object value,
                int index, boolean isSelected, boolean cellHasFocus) {
            
            super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
            
            if (value instanceof ListItem) {
                ListItem item = (ListItem) value;
                
                setText("<html><b>" + item.getTitle() + "</b><br>" +
                       "<small>" + item.getDescription() + "</small></html>");
                
                if (item.isHighPriority()) {
                    setIcon(createPriorityIcon(Color.RED));
                } else {
                    setIcon(createPriorityIcon(Color.GRAY));
                }
                
                if (isSelected) {
                    setBackground(list.getSelectionBackground());
                    setForeground(list.getSelectionForeground());
                } else {
                    setBackground(list.getBackground());
                    setForeground(list.getForeground());
                }
            }
            
            return this;
        }
        
        private Icon createPriorityIcon(Color color) {
            return new Icon() {
                @Override
                public void paintIcon(Component c, Graphics g, int x, int y) {
                    g.setColor(color);
                    g.fillOval(x, y + 2, 8, 8);
                    g.setColor(Color.BLACK);
                    g.drawOval(x, y + 2, 8, 8);
                }
                
                @Override
                public int getIconWidth() { return 10; }
                
                @Override
                public int getIconHeight() { return 12; }
            };
        }
    }
    
    private void configureFrame() {
        setTitle("JList Demo");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JListDemo().setVisible(true);
        });
    }
}
```

## 3. JTable - Tabular Data Display

JTable displays data in a tabular format with rows and columns.

### Basic JTable Example
```java
import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableColumn;
import javax.swing.table.TableRowSorter;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.text.DecimalFormat;

public class JTableDemo extends JFrame {
    private JTable table;
    private DefaultTableModel tableModel;
    private JLabel statusLabel;
    
    public JTableDemo() {
        setupTable();
        configureFrame();
    }
    
    private void setupTable() {
        setLayout(new BorderLayout());
        
        // Create table model
        String[] columnNames = {"ID", "Name", "Department", "Salary", "Years", "Active"};
        Object[][] data = {
            {1, "John Smith", "Engineering", 75000.0, 5, true},
            {2, "Jane Doe", "Marketing", 65000.0, 3, true},
            {3, "Bob Johnson", "Sales", 55000.0, 8, false},
            {4, "Alice Brown", "Engineering", 82000.0, 7, true},
            {5, "Charlie Wilson", "HR", 60000.0, 4, true},
            {6, "Diana Lee", "Finance", 70000.0, 6, true},
            {7, "Edward Davis", "Sales", 58000.0, 2, true},
            {8, "Fiona Miller", "Marketing", 67000.0, 4, false}
        };
        
        tableModel = new DefaultTableModel(data, columnNames) {
            @Override
            public Class<?> getColumnClass(int column) {
                switch (column) {
                    case 0: return Integer.class;
                    case 1: case 2: return String.class;
                    case 3: return Double.class;
                    case 4: return Integer.class;
                    case 5: return Boolean.class;
                    default: return Object.class;
                }
            }
            
            @Override
            public boolean isCellEditable(int row, int column) {
                return column != 0; // ID column is not editable
            }
        };
        
        table = new JTable(tableModel);
        
        // Enable sorting
        TableRowSorter<DefaultTableModel> sorter = new TableRowSorter<>(tableModel);
        table.setRowSorter(sorter);
        
        // Customize column widths
        TableColumn idColumn = table.getColumnModel().getColumn(0);
        idColumn.setPreferredWidth(50);
        idColumn.setMaxWidth(50);
        
        TableColumn nameColumn = table.getColumnModel().getColumn(1);
        nameColumn.setPreferredWidth(120);
        
        TableColumn deptColumn = table.getColumnModel().getColumn(2);
        deptColumn.setPreferredWidth(100);
        
        TableColumn salaryColumn = table.getColumnModel().getColumn(3);
        salaryColumn.setPreferredWidth(80);
        
        TableColumn yearsColumn = table.getColumnModel().getColumn(4);
        yearsColumn.setPreferredWidth(60);
        
        TableColumn activeColumn = table.getColumnModel().getColumn(5);
        activeColumn.setPreferredWidth(60);
        
        // Custom cell renderer for salary column
        salaryColumn.setCellRenderer(new SalaryRenderer());
        
        // Add selection listener
        table.getSelectionModel().addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                updateStatusLabel();
            }
        });
        
        // Add double-click handler
        table.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) {
                    int row = table.getSelectedRow();
                    if (row != -1) {
                        showEmployeeDetails(row);
                    }
                }
            }
        });
        
        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Employee Data"));
        
        // Control panel
        JPanel controlPanel = new JPanel(new FlowLayout());
        
        JButton addButton = new JButton("Add Employee");
        addButton.addActionListener(e -> addEmployee());
        
        JButton removeButton = new JButton("Remove Selected");
        removeButton.addActionListener(e -> removeSelectedEmployee());
        
        JButton editButton = new JButton("Edit Selected");
        editButton.addActionListener(e -> editSelectedEmployee());
        
        JTextField searchField = new JTextField(15);
        searchField.addActionListener(e -> filterTable(searchField.getText()));
        
        JButton searchButton = new JButton("Search");
        searchButton.addActionListener(e -> filterTable(searchField.getText()));
        
        JButton clearFilterButton = new JButton("Clear Filter");
        clearFilterButton.addActionListener(e -> {
            searchField.setText("");
            sorter.setRowFilter(null);
            statusLabel.setText("Filter cleared");
        });
        
        controlPanel.add(new JLabel("Search:"));
        controlPanel.add(searchField);
        controlPanel.add(searchButton);
        controlPanel.add(clearFilterButton);
        controlPanel.add(new JSeparator(SwingConstants.VERTICAL));
        controlPanel.add(addButton);
        controlPanel.add(editButton);
        controlPanel.add(removeButton);
        
        // Status label
        statusLabel = new JLabel("Click on rows to select, double-click for details");
        statusLabel.setBorder(BorderFactory.createLoweredBevelBorder());
        
        add(controlPanel, BorderLayout.NORTH);
        add(scrollPane, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
        
        updateStatusLabel();
    }
    
    private void filterTable(String searchText) {
        TableRowSorter<DefaultTableModel> sorter = 
            (TableRowSorter<DefaultTableModel>) table.getRowSorter();
        
        if (searchText.trim().isEmpty()) {
            sorter.setRowFilter(null);
            statusLabel.setText("Filter cleared");
        } else {
            try {
                sorter.setRowFilter(RowFilter.regexFilter("(?i)" + searchText));
                statusLabel.setText("Filtered by: " + searchText + 
                                   " (" + table.getRowCount() + " rows shown)");
            } catch (java.util.regex.PatternSyntaxException e) {
                statusLabel.setText("Invalid search pattern");
            }
        }
    }
    
    private void updateStatusLabel() {
        int selectedRows = table.getSelectedRowCount();
        int totalRows = table.getRowCount();
        
        if (selectedRows == 0) {
            statusLabel.setText("No selection (" + totalRows + " total rows)");
        } else if (selectedRows == 1) {
            int row = table.getSelectedRow();
            String name = (String) table.getValueAt(row, 1);
            statusLabel.setText("Selected: " + name + " (" + totalRows + " total rows)");
        } else {
            statusLabel.setText(selectedRows + " rows selected (" + totalRows + " total rows)");
        }
    }
    
    private void showEmployeeDetails(int row) {
        int modelRow = table.convertRowIndexToModel(row);
        
        StringBuilder details = new StringBuilder("Employee Details:\n\n");
        for (int col = 0; col < table.getColumnCount(); col++) {
            String columnName = table.getColumnName(col);
            Object value = tableModel.getValueAt(modelRow, col);
            details.append(columnName).append(": ").append(value).append("\n");
        }
        
        JOptionPane.showMessageDialog(this, details.toString(), 
                                     "Employee Details", JOptionPane.INFORMATION_MESSAGE);
    }
    
    private void addEmployee() {
        // Simple dialog for adding employee
        JTextField nameField = new JTextField();
        JTextField deptField = new JTextField();
        JTextField salaryField = new JTextField();
        JTextField yearsField = new JTextField();
        JCheckBox activeBox = new JCheckBox("Active", true);
        
        Object[] message = {
            "Name:", nameField,
            "Department:", deptField,
            "Salary:", salaryField,
            "Years of Service:", yearsField,
            activeBox
        };
        
        int option = JOptionPane.showConfirmDialog(this, message, 
                                                  "Add New Employee", JOptionPane.OK_CANCEL_OPTION);
        
        if (option == JOptionPane.OK_OPTION) {
            try {
                int newId = tableModel.getRowCount() + 1;
                String name = nameField.getText().trim();
                String dept = deptField.getText().trim();
                double salary = Double.parseDouble(salaryField.getText());
                int years = Integer.parseInt(yearsField.getText());
                boolean active = activeBox.isSelected();
                
                if (name.isEmpty() || dept.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Name and Department are required!");
                    return;
                }
                
                tableModel.addRow(new Object[]{newId, name, dept, salary, years, active});
                statusLabel.setText("Employee added: " + name);
                
            } catch (NumberFormatException e) {
                JOptionPane.showMessageDialog(this, "Invalid number format!");
            }
        }
    }
    
    private void removeSelectedEmployee() {
        int selectedRow = table.getSelectedRow();
        if (selectedRow == -1) {
            JOptionPane.showMessageDialog(this, "Please select an employee to remove.");
            return;
        }
        
        int modelRow = table.convertRowIndexToModel(selectedRow);
        String name = (String) tableModel.getValueAt(modelRow, 1);
        
        int option = JOptionPane.showConfirmDialog(this,
                "Are you sure you want to remove " + name + "?",
                "Confirm Removal", JOptionPane.YES_NO_OPTION);
        
        if (option == JOptionPane.YES_OPTION) {
            tableModel.removeRow(modelRow);
            statusLabel.setText("Employee removed: " + name);
        }
    }
    
    private void editSelectedEmployee() {
        int selectedRow = table.getSelectedRow();
        if (selectedRow == -1) {
            JOptionPane.showMessageDialog(this, "Please select an employee to edit.");
            return;
        }
        
        // Start editing the selected cell
        table.editCellAt(selectedRow, 1); // Start with name column
        table.getEditorComponent().requestFocus();
        statusLabel.setText("Editing mode - press Enter to confirm changes");
    }
    
    // Custom renderer for salary column
    private class SalaryRenderer extends DefaultTableCellRenderer {
        private DecimalFormat formatter = new DecimalFormat("$#,##0.00");
        
        @Override
        public Component getTableCellRendererComponent(JTable table, Object value,
                boolean isSelected, boolean hasFocus, int row, int column) {
            
            super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
            
            if (value instanceof Double) {
                double salary = (Double) value;
                setText(formatter.format(salary));
                setHorizontalAlignment(SwingConstants.RIGHT);
                
                // Color code based on salary range
                if (!isSelected) {
                    if (salary >= 80000) {
                        setBackground(new Color(200, 255, 200)); // Light green
                    } else if (salary >= 65000) {
                        setBackground(new Color(255, 255, 200)); // Light yellow
                    } else {
                        setBackground(new Color(255, 220, 220)); // Light red
                    }
                }
            }
            
            return this;
        }
    }
    
    private void configureFrame() {
        setTitle("JTable Demo - Employee Management");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 400);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JTableDemo().setVisible(true);
        });
    }
}
```

## 4. JTree - Hierarchical Data Display

JTree displays hierarchical data in a tree structure.

### JTree Example
```java
import javax.swing.*;
import javax.swing.tree.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Enumeration;

public class JTreeDemo extends JFrame {
    private JTree tree;
    private DefaultTreeModel treeModel;
    private JLabel statusLabel;
    private JTextArea detailsArea;
    
    public JTreeDemo() {
        setupTree();
        configureFrame();
    }
    
    private void setupTree() {
        setLayout(new BorderLayout());
        
        // Create tree structure
        DefaultMutableTreeNode root = new DefaultMutableTreeNode("Company");
        
        // Engineering department
        DefaultMutableTreeNode engineering = new DefaultMutableTreeNode("Engineering");
        engineering.add(new DefaultMutableTreeNode("Software Development"));
        engineering.add(new DefaultMutableTreeNode("Quality Assurance"));
        engineering.add(new DefaultMutableTreeNode("DevOps"));
        root.add(engineering);
        
        // Sales department
        DefaultMutableTreeNode sales = new DefaultMutableTreeNode("Sales");
        sales.add(new DefaultMutableTreeNode("Inside Sales"));
        sales.add(new DefaultMutableTreeNode("Field Sales"));
        sales.add(new DefaultMutableTreeNode("Sales Support"));
        root.add(sales);
        
        // Marketing department
        DefaultMutableTreeNode marketing = new DefaultMutableTreeNode("Marketing");
        marketing.add(new DefaultMutableTreeNode("Digital Marketing"));
        marketing.add(new DefaultMutableTreeNode("Content Marketing"));
        marketing.add(new DefaultMutableTreeNode("Product Marketing"));
        root.add(marketing);
        
        // Human Resources
        DefaultMutableTreeNode hr = new DefaultMutableTreeNode("Human Resources");
        hr.add(new DefaultMutableTreeNode("Recruiting"));
        hr.add(new DefaultMutableTreeNode("Employee Relations"));
        hr.add(new DefaultMutableTreeNode("Benefits"));
        root.add(hr);
        
        // Create tree model and tree
        treeModel = new DefaultTreeModel(root);
        tree = new JTree(treeModel);
        
        // Customize tree appearance
        tree.setShowsRootHandles(true);
        tree.setRootVisible(true);
        tree.setEditable(true);
        
        // Custom tree cell renderer
        tree.setCellRenderer(new CustomTreeCellRenderer());
        
        // Add selection listener
        tree.addTreeSelectionListener(e -> {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) 
                tree.getLastSelectedPathComponent();
            updateDetails(node);
        });
        
        // Add double-click listener for expansion/collapse
        tree.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) {
                    TreePath path = tree.getPathForLocation(e.getX(), e.getY());
                    if (path != null) {
                        if (tree.isExpanded(path)) {
                            tree.collapsePath(path);
                        } else {
                            tree.expandPath(path);
                        }
                    }
                }
            }
        });
        
        // Add right-click context menu
        tree.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (SwingUtilities.isRightMouseButton(e)) {
                    TreePath path = tree.getPathForLocation(e.getX(), e.getY());
                    if (path != null) {
                        tree.setSelectionPath(path);
                        showContextMenu(e.getX(), e.getY());
                    }
                }
            }
        });
        
        JScrollPane treeScrollPane = new JScrollPane(tree);
        treeScrollPane.setBorder(BorderFactory.createTitledBorder("Organization Tree"));
        treeScrollPane.setPreferredSize(new Dimension(300, 400));
        
        // Details panel
        detailsArea = new JTextArea(10, 30);
        detailsArea.setEditable(false);
        detailsArea.setFont(new Font("Courier New", Font.PLAIN, 12));
        detailsArea.setText("Select a node to see details...");
        
        JScrollPane detailsScrollPane = new JScrollPane(detailsArea);
        detailsScrollPane.setBorder(BorderFactory.createTitledBorder("Node Details"));
        
        // Control panel
        JPanel controlPanel = new JPanel(new FlowLayout());
        
        JButton expandAllButton = new JButton("Expand All");
        expandAllButton.addActionListener(e -> expandAll());
        
        JButton collapseAllButton = new JButton("Collapse All");
        collapseAllButton.addActionListener(e -> collapseAll());
        
        JButton addNodeButton = new JButton("Add Child Node");
        addNodeButton.addActionListener(e -> addChildNode());
        
        JButton removeNodeButton = new JButton("Remove Node");
        removeNodeButton.addActionListener(e -> removeSelectedNode());
        
        JButton searchButton = new JButton("Search");
        searchButton.addActionListener(e -> searchInTree());
        
        controlPanel.add(expandAllButton);
        controlPanel.add(collapseAllButton);
        controlPanel.add(new JSeparator(SwingConstants.VERTICAL));
        controlPanel.add(addNodeButton);
        controlPanel.add(removeNodeButton);
        controlPanel.add(new JSeparator(SwingConstants.VERTICAL));
        controlPanel.add(searchButton);
        
        // Status label
        statusLabel = new JLabel("Ready - Right-click for context menu");
        statusLabel.setBorder(BorderFactory.createLoweredBevelBorder());
        
        // Layout
        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT,
                                             treeScrollPane, detailsScrollPane);
        splitPane.setDividerLocation(300);
        
        add(controlPanel, BorderLayout.NORTH);
        add(splitPane, BorderLayout.CENTER);
        add(statusLabel, BorderLayout.SOUTH);
        
        // Expand root and first level
        tree.expandRow(0);
        tree.expandRow(1);
    }
    
    private void updateDetails(DefaultMutableTreeNode node) {
        if (node == null) {
            detailsArea.setText("No node selected");
            statusLabel.setText("No selection");
            return;
        }
        
        StringBuilder details = new StringBuilder();
        details.append("Node Information:\n");
        details.append("================\n\n");
        details.append("Name: ").append(node.getUserObject()).append("\n");
        details.append("Level: ").append(node.getLevel()).append("\n");
        details.append("Children: ").append(node.getChildCount()).append("\n");
        details.append("Is Leaf: ").append(node.isLeaf()).append("\n");
        details.append("Is Root: ").append(node.isRoot()).append("\n\n");
        
        if (node.getParent() != null) {
            details.append("Parent: ").append(node.getParent()).append("\n");
        }
        
        if (node.getChildCount() > 0) {
            details.append("\nChildren:\n");
            for (int i = 0; i < node.getChildCount(); i++) {
                TreeNode child = node.getChildAt(i);
                details.append("  - ").append(child).append("\n");
            }
        }
        
        // Path to root
        TreeNode[] path = node.getPath();
        details.append("\nPath to Root:\n");
        for (int i = 0; i < path.length; i++) {
            details.append("  ").append(i).append(": ").append(path[i]).append("\n");
        }
        
        detailsArea.setText(details.toString());
        statusLabel.setText("Selected: " + node.getUserObject());
    }
    
    private void showContextMenu(int x, int y) {
        JPopupMenu popup = new JPopupMenu();
        
        JMenuItem addChild = new JMenuItem("Add Child");
        addChild.addActionListener(e -> addChildNode());
        
        JMenuItem rename = new JMenuItem("Rename");
        rename.addActionListener(e -> tree.startEditingAtPath(tree.getSelectionPath()));
        
        JMenuItem remove = new JMenuItem("Remove");
        remove.addActionListener(e -> removeSelectedNode());
        
        JMenuItem expand = new JMenuItem("Expand");
        expand.addActionListener(e -> tree.expandPath(tree.getSelectionPath()));
        
        JMenuItem collapse = new JMenuItem("Collapse");
        collapse.addActionListener(e -> tree.collapsePath(tree.getSelectionPath()));
        
        popup.add(addChild);
        popup.add(rename);
        popup.addSeparator();
        popup.add(expand);
        popup.add(collapse);
        popup.addSeparator();
        popup.add(remove);
        
        popup.show(tree, x, y);
    }
    
    private void expandAll() {
        for (int i = 0; i < tree.getRowCount(); i++) {
            tree.expandRow(i);
        }
        statusLabel.setText("All nodes expanded");
    }
    
    private void collapseAll() {
        for (int i = tree.getRowCount() - 1; i >= 1; i--) {
            tree.collapseRow(i);
        }
        statusLabel.setText("All nodes collapsed");
    }
    
    private void addChildNode() {
        DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) 
            tree.getLastSelectedPathComponent();
        
        if (selectedNode == null) {
            JOptionPane.showMessageDialog(this, "Please select a parent node first.");
            return;
        }
        
        String nodeName = JOptionPane.showInputDialog(this, 
            "Enter name for new child node:", "Add Child Node", JOptionPane.QUESTION_MESSAGE);
        
        if (nodeName != null && !nodeName.trim().isEmpty()) {
            DefaultMutableTreeNode newNode = new DefaultMutableTreeNode(nodeName.trim());
            selectedNode.add(newNode);
            treeModel.nodeStructureChanged(selectedNode);
            
            // Expand parent and select new node
            tree.expandPath(new TreePath(selectedNode.getPath()));
            tree.setSelectionPath(new TreePath(newNode.getPath()));
            
            statusLabel.setText("Added child node: " + nodeName);
        }
    }
    
    private void removeSelectedNode() {
        DefaultMutableTreeNode selectedNode = (DefaultMutableTreeNode) 
            tree.getLastSelectedPathComponent();
        
        if (selectedNode == null) {
            JOptionPane.showMessageDialog(this, "Please select a node to remove.");
            return;
        }
        
        if (selectedNode.isRoot()) {
            JOptionPane.showMessageDialog(this, "Cannot remove the root node.");
            return;
        }
        
        int option = JOptionPane.showConfirmDialog(this,
            "Are you sure you want to remove '" + selectedNode.getUserObject() + "'?",
            "Confirm Removal", JOptionPane.YES_NO_OPTION);
        
        if (option == JOptionPane.YES_OPTION) {
            DefaultMutableTreeNode parent = (DefaultMutableTreeNode) selectedNode.getParent();
            parent.remove(selectedNode);
            treeModel.nodeStructureChanged(parent);
            statusLabel.setText("Node removed: " + selectedNode.getUserObject());
        }
    }
    
    private void searchInTree() {
        String searchTerm = JOptionPane.showInputDialog(this, 
            "Enter search term:", "Search Tree", JOptionPane.QUESTION_MESSAGE);
        
        if (searchTerm == null || searchTerm.trim().isEmpty()) {
            return;
        }
        
        DefaultMutableTreeNode root = (DefaultMutableTreeNode) treeModel.getRoot();
        DefaultMutableTreeNode found = searchNode(root, searchTerm.trim());
        
        if (found != null) {
            TreePath path = new TreePath(found.getPath());
            tree.setSelectionPath(path);
            tree.scrollPathToVisible(path);
            statusLabel.setText("Found: " + found.getUserObject());
        } else {
            JOptionPane.showMessageDialog(this, "Node not found: " + searchTerm);
            statusLabel.setText("Search completed - not found");
        }
    }
    
    private DefaultMutableTreeNode searchNode(DefaultMutableTreeNode node, String searchTerm) {
        if (node.getUserObject().toString().toLowerCase().contains(searchTerm.toLowerCase())) {
            return node;
        }
        
        Enumeration<TreeNode> children = node.children();
        while (children.hasMoreElements()) {
            DefaultMutableTreeNode child = (DefaultMutableTreeNode) children.nextElement();
            DefaultMutableTreeNode result = searchNode(child, searchTerm);
            if (result != null) {
                return result;
            }
        }
        
        return null;
    }
    
    // Custom tree cell renderer
    private class CustomTreeCellRenderer extends DefaultTreeCellRenderer {
        @Override
        public Component getTreeCellRendererComponent(JTree tree, Object value,
                boolean selected, boolean expanded, boolean leaf, int row, boolean hasFocus) {
            
            super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);
            
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) value;
            String nodeText = node.getUserObject().toString();
            
            // Set different icons based on node type
            if (node.isRoot()) {
                setIcon(createColorIcon(Color.BLUE));
            } else if (node.isLeaf()) {
                setIcon(createColorIcon(Color.GREEN));
            } else {
                setIcon(createColorIcon(Color.ORANGE));
            }
            
            // Add child count to non-leaf nodes
            if (!node.isLeaf()) {
                setText(nodeText + " (" + node.getChildCount() + ")");
            }
            
            return this;
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
    }
    
    private void configureFrame() {
        setTitle("JTree Demo - Organization Structure");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 500);
        setLocationRelativeTo(null);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new JTreeDemo().setVisible(true);
        });
    }
}
```

## Summary

Advanced Swing components provide sophisticated data presentation and interaction capabilities:

### Component Capabilities
- **JComboBox**: Dropdown selection with editable options and auto-completion
- **JList**: Single/multiple selection lists with custom rendering
- **JTable**: Tabular data with sorting, filtering, and editing
- **JTree**: Hierarchical data navigation with context menus

### Advanced Features Demonstrated
- Custom cell renderers for enhanced visual presentation
- Data filtering and searching capabilities
- Context menus and popup interactions
- Real-time data updates and validation
- Sorting and organizing large datasets
- Editable content with immediate feedback

### Key Patterns
- Model-View separation for data management
- Custom renderers for specialized display needs
- Event handling for complex user interactions
- Data validation and error handling
- Performance optimization for large datasets

### Best Practices
- Use appropriate selection modes for user workflows
- Implement custom renderers for better visual communication
- Provide search and filter capabilities for large datasets
- Add context menus for advanced operations
- Handle data persistence and undo operations
- Optimize performance with lazy loading and virtual scrolling
- Maintain consistent visual styling across components

These advanced components enable the creation of sophisticated business applications with rich data interaction capabilities.

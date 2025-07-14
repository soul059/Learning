# Layouts and Views

## Table of Contents
- [Introduction to Views](#introduction-to-views)
- [Layout Types](#layout-types)
- [Common Views](#common-views)
- [Layout Attributes](#layout-attributes)
- [ViewGroups](#viewgroups)
- [Creating Layouts](#creating-layouts)
- [Dynamic Layouts](#dynamic-layouts)
- [Best Practices](#best-practices)

## Introduction to Views

A **View** is the basic building block for user interface components. Everything you see on an Android screen is a view or a group of views.

### View Hierarchy
```
Activity
└── ViewGroup (Layout)
    ├── View (Button)
    ├── View (TextView)
    └── ViewGroup (LinearLayout)
        ├── View (EditText)
        └── View (ImageView)
```

### Key Concepts
- **View**: Single UI component (Button, TextView, ImageView)
- **ViewGroup**: Container that holds other views (LinearLayout, RelativeLayout)
- **Layout**: XML file that defines the structure of views

## Layout Types

### 1. LinearLayout
Arranges child views in a single row or column.

```xml
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Header"
        android:textSize="24sp"
        android:gravity="center" />

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        android:layout_marginTop="16dp">

        <Button
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:text="Button 1"
            android:layout_marginEnd="8dp" />

        <Button
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:text="Button 2"
            android:layout_marginStart="8dp" />

    </LinearLayout>

</LinearLayout>
```

**Attributes:**
- `android:orientation`: "horizontal" or "vertical"
- `android:layout_weight`: Distributes remaining space
- `android:gravity`: Aligns child views within the layout
- `android:layout_gravity`: Aligns the view within its parent

### 2. RelativeLayout
Positions child views relative to each other or the parent.

```xml
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <TextView
        android:id="@+id/titleText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Title"
        android:textSize="24sp"
        android:layout_centerHorizontal="true" />

    <EditText
        android:id="@+id/editTextName"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:hint="Enter your name"
        android:layout_below="@id/titleText"
        android:layout_marginTop="20dp" />

    <Button
        android:id="@+id/submitButton"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Submit"
        android:layout_below="@id/editTextName"
        android:layout_centerHorizontal="true"
        android:layout_marginTop="16dp" />

    <TextView
        android:id="@+id/footerText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Footer"
        android:layout_alignParentBottom="true"
        android:layout_centerHorizontal="true" />

</RelativeLayout>
```

**Common Attributes:**
- `android:layout_above/below`
- `android:layout_toLeftOf/toRightOf`
- `android:layout_alignTop/Bottom/Left/Right`
- `android:layout_centerInParent`
- `android:layout_alignParentTop/Bottom/Left/Right`

### 3. ConstraintLayout
Most flexible and efficient layout for complex UIs.

```xml
<androidx.constraintlayout.widget.ConstraintLayout 
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <TextView
        android:id="@+id/titleText"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:text="Welcome"
        android:textSize="24sp"
        android:gravity="center"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.5" />

    <EditText
        android:id="@+id/editTextEmail"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:hint="Email"
        android:inputType="textEmailAddress"
        app:layout_constraintTop_toBottomOf="@id/titleText"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="32dp" />

    <EditText
        android:id="@+id/editTextPassword"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:hint="Password"
        android:inputType="textPassword"
        app:layout_constraintTop_toBottomOf="@id/editTextEmail"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="16dp" />

    <Button
        android:id="@+id/loginButton"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:text="Login"
        app:layout_constraintTop_toBottomOf="@id/editTextPassword"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="24dp" />

    <TextView
        android:id="@+id/forgotPassword"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Forgot Password?"
        android:textColor="@color/colorPrimary"
        app:layout_constraintTop_toBottomOf="@id/loginButton"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="16dp" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

**Key Attributes:**
- `app:layout_constraintTop_toTopOf`
- `app:layout_constraintStart_toStartOf`
- `app:layout_constraintEnd_toEndOf`
- `app:layout_constraintBottom_toBottomOf`

### 4. FrameLayout
Simplest layout - stacks child views on top of each other.

```xml
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <ImageView
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:src="@drawable/background_image"
        android:scaleType="centerCrop" />

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Overlay Text"
        android:textColor="@android:color/white"
        android:textSize="24sp"
        android:layout_gravity="center" />

    <ProgressBar
        android:id="@+id/progressBar"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:visibility="gone" />

</FrameLayout>
```

### 5. TableLayout
Arranges child views in rows and columns.

```xml
<TableLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp"
    android:stretchColumns="1">

    <TableRow>
        <TextView
            android:text="Name:"
            android:textStyle="bold"
            android:padding="8dp" />
        <EditText
            android:hint="Enter name"
            android:padding="8dp" />
    </TableRow>

    <TableRow>
        <TextView
            android:text="Email:"
            android:textStyle="bold"
            android:padding="8dp" />
        <EditText
            android:hint="Enter email"
            android:inputType="textEmailAddress"
            android:padding="8dp" />
    </TableRow>

    <TableRow>
        <TextView
            android:text="Phone:"
            android:textStyle="bold"
            android:padding="8dp" />
        <EditText
            android:hint="Enter phone"
            android:inputType="phone"
            android:padding="8dp" />
    </TableRow>

</TableLayout>
```

### 6. GridLayout
Arranges child views in a rectangular grid.

```xml
<GridLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:columnCount="2"
    android:rowCount="3"
    android:padding="16dp">

    <Button
        android:text="Button 1"
        android:layout_columnWeight="1"
        android:layout_margin="4dp" />

    <Button
        android:text="Button 2"
        android:layout_columnWeight="1"
        android:layout_margin="4dp" />

    <Button
        android:text="Button 3"
        android:layout_columnSpan="2"
        android:layout_columnWeight="1"
        android:layout_margin="4dp" />

    <Button
        android:text="Button 4"
        android:layout_columnWeight="1"
        android:layout_margin="4dp" />

    <Button
        android:text="Button 5"
        android:layout_columnWeight="1"
        android:layout_margin="4dp" />

</GridLayout>
```

## Common Views

### 1. TextView
Displays text to the user.

```xml
<TextView
    android:id="@+id/textView"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Hello World!"
    android:textSize="18sp"
    android:textColor="@color/black"
    android:textStyle="bold"
    android:fontFamily="sans-serif"
    android:gravity="center"
    android:padding="16dp"
    android:background="@color/lightGray"
    android:maxLines="2"
    android:ellipsize="end" />
```

```java
// Programmatically
TextView textView = findViewById(R.id.textView);
textView.setText("New text");
textView.setTextSize(20);
textView.setTextColor(Color.BLUE);
```

### 2. EditText
Allows user to enter and edit text.

```xml
<EditText
    android:id="@+id/editText"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:hint="Enter your message"
    android:inputType="textMultiLine"
    android:lines="3"
    android:maxLength="200"
    android:padding="12dp"
    android:background="@drawable/edittext_background"
    android:textSize="16sp" />
```

**Input Types:**
- `text`: Plain text
- `textEmailAddress`: Email format
- `textPassword`: Password (hidden)
- `number`: Numeric only
- `phone`: Phone number
- `textMultiLine`: Multiple lines

```java
// Programmatically
EditText editText = findViewById(R.id.editText);
String userInput = editText.getText().toString();
editText.setText("New text");

// Add text watcher
editText.addTextChangedListener(new TextWatcher() {
    @Override
    public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

    @Override
    public void onTextChanged(CharSequence s, int start, int before, int count) {
        // Text is changing
    }

    @Override
    public void afterTextChanged(Editable s) {
        // Text has changed
        String text = s.toString();
    }
});
```

### 3. Button
Clickable button.

```xml
<Button
    android:id="@+id/button"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:text="Click Me"
    android:textSize="16sp"
    android:textColor="@android:color/white"
    android:background="@color/colorPrimary"
    android:padding="12dp"
    android:layout_margin="16dp"
    android:enabled="true" />
```

```java
// Event handling
Button button = findViewById(R.id.button);
button.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        // Handle button click
        Toast.makeText(MainActivity.this, "Button clicked!", Toast.LENGTH_SHORT).show();
    }
});

// Lambda expression (Java 8+)
button.setOnClickListener(v -> {
    // Handle click
});
```

### 4. ImageView
Displays images.

```xml
<ImageView
    android:id="@+id/imageView"
    android:layout_width="200dp"
    android:layout_height="200dp"
    android:src="@drawable/sample_image"
    android:scaleType="centerCrop"
    android:contentDescription="Sample image"
    android:background="@color/lightGray" />
```

**Scale Types:**
- `centerCrop`: Crop to fill
- `centerInside`: Scale to fit inside
- `fitXY`: Stretch to fill
- `center`: Center without scaling

```java
// Programmatically
ImageView imageView = findViewById(R.id.imageView);
imageView.setImageResource(R.drawable.new_image);
imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);

// Load from URL (using external library like Glide)
Glide.with(this)
    .load("https://example.com/image.jpg")
    .into(imageView);
```

### 5. CheckBox
Boolean selection.

```xml
<CheckBox
    android:id="@+id/checkBox"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="I agree to terms and conditions"
    android:textSize="14sp"
    android:checked="false" />
```

```java
CheckBox checkBox = findViewById(R.id.checkBox);
checkBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if (isChecked) {
            // Checkbox is checked
        } else {
            // Checkbox is unchecked
        }
    }
});

// Check programmatically
boolean isChecked = checkBox.isChecked();
checkBox.setChecked(true);
```

### 6. RadioButton and RadioGroup
Single selection from multiple options.

```xml
<RadioGroup
    android:id="@+id/radioGroup"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:orientation="vertical">

    <RadioButton
        android:id="@+id/radioOption1"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Option 1" />

    <RadioButton
        android:id="@+id/radioOption2"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Option 2" />

    <RadioButton
        android:id="@+id/radioOption3"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Option 3" />

</RadioGroup>
```

```java
RadioGroup radioGroup = findViewById(R.id.radioGroup);
radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
    @Override
    public void onCheckedChanged(RadioGroup group, int checkedId) {
        switch (checkedId) {
            case R.id.radioOption1:
                // Option 1 selected
                break;
            case R.id.radioOption2:
                // Option 2 selected
                break;
            case R.id.radioOption3:
                // Option 3 selected
                break;
        }
    }
});
```

### 7. Spinner (Dropdown)
Dropdown selection.

```xml
<Spinner
    android:id="@+id/spinner"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:layout_margin="16dp" />
```

```java
Spinner spinner = findViewById(R.id.spinner);

// Create adapter
String[] items = {"Item 1", "Item 2", "Item 3", "Item 4"};
ArrayAdapter<String> adapter = new ArrayAdapter<>(this, 
    android.R.layout.simple_spinner_item, items);
adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

// Set adapter
spinner.setAdapter(adapter);

// Handle selection
spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        String selectedItem = items[position];
        Toast.makeText(MainActivity.this, "Selected: " + selectedItem, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {}
});
```

### 8. ProgressBar
Shows progress or loading state.

```xml
<!-- Indeterminate progress -->
<ProgressBar
    android:id="@+id/progressBar"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:layout_centerInParent="true" />

<!-- Determinate progress -->
<ProgressBar
    android:id="@+id/progressBarHorizontal"
    style="@android:style/Widget.ProgressBar.Horizontal"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:max="100"
    android:progress="50" />
```

```java
ProgressBar progressBar = findViewById(R.id.progressBarHorizontal);

// Update progress
progressBar.setProgress(75);

// Show/hide loading
ProgressBar loadingBar = findViewById(R.id.progressBar);
loadingBar.setVisibility(View.VISIBLE); // Show
loadingBar.setVisibility(View.GONE);    // Hide
```

## Layout Attributes

### Size Attributes
```xml
<!-- Width and Height -->
android:layout_width="match_parent"    <!-- Fill parent -->
android:layout_width="wrap_content"    <!-- Fit content -->
android:layout_width="200dp"           <!-- Fixed size -->

<!-- Minimum size -->
android:minWidth="100dp"
android:minHeight="50dp"

<!-- Maximum size -->
android:maxWidth="300dp"
android:maxHeight="200dp"
```

### Margin and Padding
```xml
<!-- Margin (outside spacing) -->
android:layout_margin="16dp"           <!-- All sides -->
android:layout_marginTop="8dp"         <!-- Top only -->
android:layout_marginStart="12dp"      <!-- Start (left in LTR) -->
android:layout_marginEnd="12dp"        <!-- End (right in LTR) -->
android:layout_marginBottom="8dp"      <!-- Bottom only -->

<!-- Padding (inside spacing) -->
android:padding="16dp"                 <!-- All sides -->
android:paddingTop="8dp"              <!-- Top only -->
android:paddingStart="12dp"           <!-- Start only -->
android:paddingEnd="12dp"             <!-- End only -->
android:paddingBottom="8dp"           <!-- Bottom only -->
```

### Visibility
```xml
android:visibility="visible"    <!-- Default, view is visible -->
android:visibility="invisible"  <!-- Hidden but takes up space -->
android:visibility="gone"       <!-- Hidden and no space -->
```

```java
// Programmatically
view.setVisibility(View.VISIBLE);
view.setVisibility(View.INVISIBLE);
view.setVisibility(View.GONE);
```

## ViewGroups

### ScrollView
Makes content scrollable when it exceeds screen size.

```xml
<ScrollView
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:fillViewport="true">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical"
        android:padding="16dp">

        <!-- Long content here -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="Very long text content that needs scrolling..." />

        <!-- More views -->

    </LinearLayout>

</ScrollView>
```

### HorizontalScrollView
Horizontal scrolling.

```xml
<HorizontalScrollView
    android:layout_width="match_parent"
    android:layout_height="wrap_content">

    <LinearLayout
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:orientation="horizontal">

        <!-- Wide content here -->

    </LinearLayout>

</HorizontalScrollView>
```

## Creating Layouts

### XML Layout Example
```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <com.google.android.material.textfield.TextInputLayout
        android:id="@+id/textInputName"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:hint="Full Name"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent">

        <com.google.android.material.textfield.TextInputEditText
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:inputType="textPersonName" />

    </com.google.android.material.textfield.TextInputLayout>

    <com.google.android.material.button.MaterialButton
        android:id="@+id/submitButton"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:text="Submit"
        app:layout_constraintTop_toBottomOf="@id/textInputName"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="24dp" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

## Dynamic Layouts

### Creating Views Programmatically
```java
public class MainActivity extends AppCompatActivity {
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Create layout programmatically
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(32, 32, 32, 32);
        
        // Create TextView
        TextView textView = new TextView(this);
        textView.setText("Dynamic TextView");
        textView.setTextSize(18);
        textView.setGravity(Gravity.CENTER);
        
        // Create Button
        Button button = new Button(this);
        button.setText("Dynamic Button");
        button.setOnClickListener(v -> {
            Toast.makeText(this, "Dynamic button clicked!", Toast.LENGTH_SHORT).show();
        });
        
        // Add views to layout
        layout.addView(textView);
        layout.addView(button);
        
        // Set as content view
        setContentView(layout);
    }
}
```

### Adding Views to Existing Layout
```java
// Get existing layout
LinearLayout parentLayout = findViewById(R.id.parentLayout);

// Create new view
TextView dynamicText = new TextView(this);
dynamicText.setText("Added dynamically");
dynamicText.setTextSize(16);

// Set layout parameters
LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
    LinearLayout.LayoutParams.MATCH_PARENT,
    LinearLayout.LayoutParams.WRAP_CONTENT
);
params.setMargins(0, 16, 0, 0);
dynamicText.setLayoutParams(params);

// Add to parent
parentLayout.addView(dynamicText);

// Remove view
parentLayout.removeView(dynamicText);
```

## Best Practices

### 1. Use ConstraintLayout
- More efficient than nested layouts
- Better performance
- Easier to create responsive designs

### 2. Avoid Deep Nesting
```xml
<!-- Bad: Deep nesting -->
<LinearLayout>
    <LinearLayout>
        <LinearLayout>
            <TextView />
        </LinearLayout>
    </LinearLayout>
</LinearLayout>

<!-- Good: Flat hierarchy -->
<ConstraintLayout>
    <TextView />
</ConstraintLayout>
```

### 3. Use Appropriate Layout
- **LinearLayout**: Simple sequential layouts
- **ConstraintLayout**: Complex layouts
- **FrameLayout**: Overlapping views
- **RecyclerView**: Lists and grids

### 4. Optimize for Different Screen Sizes
```xml
<!-- res/layout/activity_main.xml (default) -->
<!-- res/layout-large/activity_main.xml (tablets) -->
<!-- res/layout-land/activity_main.xml (landscape) -->
```

### 5. Use Resources
```xml
<!-- Use dimension resources -->
android:padding="@dimen/standard_padding"
android:layout_margin="@dimen/standard_margin"

<!-- Use color resources -->
android:textColor="@color/primary_text"
android:background="@color/background_color"

<!-- Use string resources -->
android:text="@string/welcome_message"
android:hint="@string/enter_name"
```

### 6. Include and Merge Tags
```xml
<!-- reusable_layout.xml -->
<merge xmlns:android="http://schemas.android.com/apk/res/android">
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Reusable content" />
</merge>

<!-- main_layout.xml -->
<LinearLayout>
    <include layout="@layout/reusable_layout" />
</LinearLayout>
```

### 7. ViewStub for Lazy Loading
```xml
<ViewStub
    android:id="@+id/viewStub"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:layout="@layout/expensive_layout" />
```

```java
// Inflate when needed
ViewStub viewStub = findViewById(R.id.viewStub);
View inflatedView = viewStub.inflate();
```

Understanding layouts and views is fundamental to creating effective Android user interfaces. Choose the right layout type for your needs and follow best practices for optimal performance and maintainability.

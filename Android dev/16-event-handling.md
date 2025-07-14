# Event Handling in Android

## Table of Contents
- [Overview of Event Handling](#overview-of-event-handling)
- [Click Events](#click-events)
- [Touch Events](#touch-events)
- [Focus Events](#focus-events)
- [Key Events](#key-events)
- [Text Input Events](#text-input-events)
- [Gesture Events](#gesture-events)
- [Custom Event Handling](#custom-event-handling)
- [Best Practices](#best-practices)

## Overview of Event Handling

Event handling in Android refers to responding to user interactions with the UI. Events include clicks, touches, gestures, key presses, and focus changes.

### Event Flow
1. **Event Generation**: User performs an action (touch, click, key press)
2. **Event Dispatch**: Android system dispatches the event to the appropriate view
3. **Event Handling**: The view or its listeners process the event
4. **Event Consumption**: Event is marked as handled or passed to parent views

### Common Event Types
- **Click Events**: Button presses, item selections
- **Touch Events**: Raw touch interactions, gestures
- **Focus Events**: View gaining/losing focus
- **Key Events**: Hardware/software keyboard input
- **Motion Events**: Scrolling, dragging, swiping

## Click Events

### 1. Basic Button Click Handling

#### Method 1: OnClickListener Interface
```java
public class MainActivity extends AppCompatActivity {
    
    private Button submitButton;
    private Button cancelButton;
    private TextView statusText;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initializeViews();
        setupClickListeners();
    }
    
    private void initializeViews() {
        submitButton = findViewById(R.id.submitButton);
        cancelButton = findViewById(R.id.cancelButton);
        statusText = findViewById(R.id.statusText);
    }
    
    private void setupClickListeners() {
        // Method 1: Anonymous inner class
        submitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handleSubmitClick();
            }
        });
        
        // Method 2: Lambda expression (Java 8+)
        cancelButton.setOnClickListener(v -> handleCancelClick());
        
        // Method 3: Method reference
        // submitButton.setOnClickListener(this::handleSubmitClick);
    }
    
    private void handleSubmitClick() {
        statusText.setText("Submit button clicked!");
        
        // Disable button to prevent multiple clicks
        submitButton.setEnabled(false);
        
        // Perform submission logic
        performSubmission();
    }
    
    private void handleCancelClick() {
        statusText.setText("Cancel button clicked!");
        
        // Clear form or navigate back
        clearForm();
    }
    
    private void performSubmission() {
        // Simulate async operation
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            statusText.setText("Submission completed!");
            submitButton.setEnabled(true);
        }, 2000);
    }
    
    private void clearForm() {
        // Clear form fields
        EditText nameField = findViewById(R.id.nameField);
        EditText emailField = findViewById(R.id.emailField);
        
        nameField.setText("");
        emailField.setText("");
        statusText.setText("Form cleared");
    }
}
```

#### Method 2: XML onClick Attribute
```xml
<!-- res/layout/activity_main.xml -->
<Button
    android:id="@+id/xmlClickButton"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="XML Click Handler"
    android:onClick="onXmlButtonClick" />
```

```java
// In Activity
public void onXmlButtonClick(View view) {
    Toast.makeText(this, "XML onClick called!", Toast.LENGTH_SHORT).show();
    
    // You can also check which view was clicked
    switch (view.getId()) {
        case R.id.xmlClickButton:
            handleXmlButtonClick();
            break;
        // Add more cases for other buttons
    }
}

private void handleXmlButtonClick() {
    Log.d("MainActivity", "XML button clicked");
}
```

#### Method 3: Activity Implementing OnClickListener
```java
public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        Button button1 = findViewById(R.id.button1);
        Button button2 = findViewById(R.id.button2);
        Button button3 = findViewById(R.id.button3);
        
        // Set the same listener for multiple buttons
        button1.setOnClickListener(this);
        button2.setOnClickListener(this);
        button3.setOnClickListener(this);
    }
    
    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.button1:
                handleButton1Click();
                break;
            case R.id.button2:
                handleButton2Click();
                break;
            case R.id.button3:
                handleButton3Click();
                break;
            default:
                Log.w("MainActivity", "Unknown button clicked: " + v.getId());
        }
    }
    
    private void handleButton1Click() {
        Toast.makeText(this, "Button 1 clicked", Toast.LENGTH_SHORT).show();
    }
    
    private void handleButton2Click() {
        Toast.makeText(this, "Button 2 clicked", Toast.LENGTH_SHORT).show();
    }
    
    private void handleButton3Click() {
        Toast.makeText(this, "Button 3 clicked", Toast.LENGTH_SHORT).show();
    }
}
```

### 2. Advanced Click Handling

#### Preventing Multiple Clicks
```java
public class ClickUtils {
    private static final int CLICK_DELAY = 1000; // 1 second
    private static long lastClickTime = 0;
    
    public static boolean isValidClick() {
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastClickTime > CLICK_DELAY) {
            lastClickTime = currentTime;
            return true;
        }
        return false;
    }
}

// Usage in click listener
submitButton.setOnClickListener(v -> {
    if (ClickUtils.isValidClick()) {
        handleSubmitClick();
    }
});
```

#### Long Click Events
```java
// Long click listener
imageView.setOnLongClickListener(new View.OnLongClickListener() {
    @Override
    public boolean onLongClick(View v) {
        // Show context menu or perform long click action
        showContextMenu(v);
        return true; // Event consumed
    }
});

// Lambda version
imageView.setOnLongClickListener(v -> {
    showContextMenu(v);
    return true;
});

private void showContextMenu(View view) {
    PopupMenu popup = new PopupMenu(this, view);
    popup.getMenuInflater().inflate(R.menu.context_menu, popup.getMenu());
    
    popup.setOnMenuItemClickListener(item -> {
        switch (item.getItemId()) {
            case R.id.action_edit:
                editItem();
                return true;
            case R.id.action_delete:
                deleteItem();
                return true;
            default:
                return false;
        }
    });
    
    popup.show();
}
```

### 3. RecyclerView Item Click Handling
```java
public class ItemAdapter extends RecyclerView.Adapter<ItemAdapter.ViewHolder> {
    
    private List<Item> items;
    private OnItemClickListener clickListener;
    
    public interface OnItemClickListener {
        void onItemClick(Item item, int position);
        void onItemLongClick(Item item, int position);
    }
    
    public ItemAdapter(List<Item> items, OnItemClickListener listener) {
        this.items = items;
        this.clickListener = listener;
    }
    
    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
            .inflate(R.layout.item_layout, parent, false);
        return new ViewHolder(view);
    }
    
    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {
        Item item = items.get(position);
        holder.bind(item, position);
    }
    
    @Override
    public int getItemCount() {
        return items.size();
    }
    
    class ViewHolder extends RecyclerView.ViewHolder {
        TextView titleText;
        TextView descriptionText;
        ImageView iconImage;
        
        ViewHolder(View itemView) {
            super(itemView);
            titleText = itemView.findViewById(R.id.titleText);
            descriptionText = itemView.findViewById(R.id.descriptionText);
            iconImage = itemView.findViewById(R.id.iconImage);
            
            // Set click listeners on the entire item view
            itemView.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (position != RecyclerView.NO_POSITION && clickListener != null) {
                    clickListener.onItemClick(items.get(position), position);
                }
            });
            
            itemView.setOnLongClickListener(v -> {
                int position = getAdapterPosition();
                if (position != RecyclerView.NO_POSITION && clickListener != null) {
                    clickListener.onItemLongClick(items.get(position), position);
                }
                return true;
            });
        }
        
        void bind(Item item, int position) {
            titleText.setText(item.getTitle());
            descriptionText.setText(item.getDescription());
            
            // Load image with Glide or Picasso
            Glide.with(itemView.getContext())
                .load(item.getImageUrl())
                .into(iconImage);
        }
    }
}

// Usage in Activity/Fragment
private void setupRecyclerView() {
    RecyclerView recyclerView = findViewById(R.id.recyclerView);
    
    ItemAdapter adapter = new ItemAdapter(items, new ItemAdapter.OnItemClickListener() {
        @Override
        public void onItemClick(Item item, int position) {
            // Handle item click
            Intent intent = new Intent(MainActivity.this, DetailActivity.class);
            intent.putExtra("item_id", item.getId());
            startActivity(intent);
        }
        
        @Override
        public void onItemLongClick(Item item, int position) {
            // Handle long click - show options
            showItemOptions(item, position);
        }
    });
    
    recyclerView.setAdapter(adapter);
    recyclerView.setLayoutManager(new LinearLayoutManager(this));
}
```

## Touch Events

### 1. Basic Touch Handling
```java
public class TouchView extends View {
    
    private Paint paint;
    private Path path;
    
    public TouchView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        paint = new Paint();
        paint.setColor(Color.BLUE);
        paint.setStrokeWidth(5f);
        paint.setStyle(Paint.Style.STROKE);
        
        path = new Path();
    }
    
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();
        
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                handleTouchDown(x, y);
                return true;
                
            case MotionEvent.ACTION_MOVE:
                handleTouchMove(x, y);
                break;
                
            case MotionEvent.ACTION_UP:
                handleTouchUp(x, y);
                break;
                
            case MotionEvent.ACTION_CANCEL:
                handleTouchCancel();
                break;
        }
        
        invalidate(); // Trigger redraw
        return true;
    }
    
    private void handleTouchDown(float x, float y) {
        path.moveTo(x, y);
        Log.d("TouchView", "Touch down at: " + x + ", " + y);
    }
    
    private void handleTouchMove(float x, float y) {
        path.lineTo(x, y);
        Log.d("TouchView", "Touch move to: " + x + ", " + y);
    }
    
    private void handleTouchUp(float x, float y) {
        Log.d("TouchView", "Touch up at: " + x + ", " + y);
        // Perform final action
    }
    
    private void handleTouchCancel() {
        Log.d("TouchView", "Touch cancelled");
        path.reset();
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(path, paint);
    }
}
```

### 2. Multi-Touch Handling
```java
public class MultiTouchView extends View {
    
    private static final int MAX_POINTERS = 10;
    private Paint[] paints = new Paint[MAX_POINTERS];
    private Path[] paths = new Path[MAX_POINTERS];
    
    public MultiTouchView(Context context, AttributeSet attrs) {
        super(context, attrs);
        initPaints();
    }
    
    private void initPaints() {
        int[] colors = {Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, 
                       Color.MAGENTA, Color.CYAN, Color.BLACK, Color.GRAY,
                       Color.DKGRAY, Color.LTGRAY};
        
        for (int i = 0; i < MAX_POINTERS; i++) {
            paints[i] = new Paint();
            paints[i].setColor(colors[i % colors.length]);
            paints[i].setStrokeWidth(8f);
            paints[i].setStyle(Paint.Style.STROKE);
            paints[i].setAntiAlias(true);
            
            paths[i] = new Path();
        }
    }
    
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        int pointerCount = event.getPointerCount();
        int action = event.getActionMasked();
        int pointerIndex = event.getActionIndex();
        int pointerId = event.getPointerId(pointerIndex);
        
        switch (action) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN:
                if (pointerId < MAX_POINTERS) {
                    float x = event.getX(pointerIndex);
                    float y = event.getY(pointerIndex);
                    paths[pointerId].moveTo(x, y);
                    Log.d("MultiTouch", "Pointer " + pointerId + " down at: " + x + ", " + y);
                }
                break;
                
            case MotionEvent.ACTION_MOVE:
                for (int i = 0; i < pointerCount; i++) {
                    int id = event.getPointerId(i);
                    if (id < MAX_POINTERS) {
                        float x = event.getX(i);
                        float y = event.getY(i);
                        paths[id].lineTo(x, y);
                    }
                }
                break;
                
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP:
                Log.d("MultiTouch", "Pointer " + pointerId + " up");
                break;
                
            case MotionEvent.ACTION_CANCEL:
                clearPaths();
                break;
        }
        
        invalidate();
        return true;
    }
    
    private void clearPaths() {
        for (Path path : paths) {
            path.reset();
        }
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        for (int i = 0; i < MAX_POINTERS; i++) {
            canvas.drawPath(paths[i], paints[i]);
        }
    }
}
```

### 3. Touch Event Delegation
```java
public class TouchDelegatingActivity extends AppCompatActivity {
    
    private View parentView;
    private Button childButton;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_touch_delegating);
        
        parentView = findViewById(R.id.parentView);
        childButton = findViewById(R.id.childButton);
        
        setupTouchDelegation();
    }
    
    private void setupTouchDelegation() {
        // Expand the touch area of the button
        parentView.post(() -> {
            Rect delegateArea = new Rect();
            childButton.getHitRect(delegateArea);
            
            // Expand the touch area by 50dp in all directions
            int expansion = (int) TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, 50, getResources().getDisplayMetrics());
            
            delegateArea.top -= expansion;
            delegateArea.bottom += expansion;
            delegateArea.left -= expansion;
            delegateArea.right += expansion;
            
            TouchDelegate touchDelegate = new TouchDelegate(delegateArea, childButton);
            parentView.setTouchDelegate(touchDelegate);
        });
        
        childButton.setOnClickListener(v -> {
            Toast.makeText(this, "Button clicked with expanded touch area!", 
                         Toast.LENGTH_SHORT).show();
        });
    }
}
```

## Focus Events

### 1. Focus Change Handling
```java
public class FocusHandlingActivity extends AppCompatActivity {
    
    private EditText nameField;
    private EditText emailField;
    private EditText phoneField;
    private TextView focusIndicator;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_focus_handling);
        
        initializeViews();
        setupFocusListeners();
    }
    
    private void initializeViews() {
        nameField = findViewById(R.id.nameField);
        emailField = findViewById(R.id.emailField);
        phoneField = findViewById(R.id.phoneField);
        focusIndicator = findViewById(R.id.focusIndicator);
    }
    
    private void setupFocusListeners() {
        View.OnFocusChangeListener focusListener = new View.OnFocusChangeListener() {
            @Override
            public void onFocusChange(View v, boolean hasFocus) {
                handleFocusChange(v, hasFocus);
            }
        };
        
        nameField.setOnFocusChangeListener(focusListener);
        emailField.setOnFocusChangeListener(focusListener);
        phoneField.setOnFocusChangeListener(focusListener);
        
        // Request focus programmatically
        nameField.requestFocus();
    }
    
    private void handleFocusChange(View view, boolean hasFocus) {
        if (hasFocus) {
            String fieldName = getFieldName(view);
            focusIndicator.setText("Currently focused: " + fieldName);
            
            // Highlight the focused field
            view.setBackgroundResource(R.drawable.focused_field_background);
            
            // Perform field-specific actions
            switch (view.getId()) {
                case R.id.nameField:
                    showNameFieldHelp();
                    break;
                case R.id.emailField:
                    showEmailFieldHelp();
                    break;
                case R.id.phoneField:
                    showPhoneFieldHelp();
                    break;
            }
        } else {
            // Remove highlight when focus is lost
            view.setBackgroundResource(R.drawable.normal_field_background);
            
            // Validate field when focus is lost
            validateField(view);
        }
    }
    
    private String getFieldName(View view) {
        switch (view.getId()) {
            case R.id.nameField:
                return "Name Field";
            case R.id.emailField:
                return "Email Field";
            case R.id.phoneField:
                return "Phone Field";
            default:
                return "Unknown Field";
        }
    }
    
    private void showNameFieldHelp() {
        Toast.makeText(this, "Enter your full name", Toast.LENGTH_SHORT).show();
    }
    
    private void showEmailFieldHelp() {
        Toast.makeText(this, "Enter a valid email address", Toast.LENGTH_SHORT).show();
    }
    
    private void showPhoneFieldHelp() {
        Toast.makeText(this, "Enter your phone number", Toast.LENGTH_SHORT).show();
    }
    
    private void validateField(View view) {
        if (view instanceof EditText) {
            EditText editText = (EditText) view;
            String text = editText.getText().toString().trim();
            
            switch (view.getId()) {
                case R.id.nameField:
                    validateName(editText, text);
                    break;
                case R.id.emailField:
                    validateEmail(editText, text);
                    break;
                case R.id.phoneField:
                    validatePhone(editText, text);
                    break;
            }
        }
    }
    
    private void validateName(EditText field, String name) {
        if (name.isEmpty()) {
            field.setError("Name is required");
        } else if (name.length() < 2) {
            field.setError("Name must be at least 2 characters");
        } else {
            field.setError(null);
        }
    }
    
    private void validateEmail(EditText field, String email) {
        if (email.isEmpty()) {
            field.setError("Email is required");
        } else if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            field.setError("Please enter a valid email address");
        } else {
            field.setError(null);
        }
    }
    
    private void validatePhone(EditText field, String phone) {
        if (phone.isEmpty()) {
            field.setError("Phone number is required");
        } else if (phone.length() < 10) {
            field.setError("Please enter a valid phone number");
        } else {
            field.setError(null);
        }
    }
}
```

## Key Events

### 1. Hardware Key Handling
```java
public class KeyEventActivity extends AppCompatActivity {
    
    private TextView keyEventText;
    private StringBuilder keyLog;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_key_event);
        
        keyEventText = findViewById(R.id.keyEventText);
        keyLog = new StringBuilder();
        
        // Make activity focusable to receive key events
        getWindow().getDecorView().setFocusableInTouchMode(true);
        getWindow().getDecorView().requestFocus();
    }
    
    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        logKeyEvent("onKeyDown", keyCode, event);
        
        switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_UP:
                handleVolumeUp();
                return true; // Consume the event
                
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                handleVolumeDown();
                return true;
                
            case KeyEvent.KEYCODE_BACK:
                handleBackKey();
                return true;
                
            case KeyEvent.KEYCODE_MENU:
                handleMenuKey();
                return true;
                
            case KeyEvent.KEYCODE_SEARCH:
                handleSearchKey();
                return true;
                
            default:
                return super.onKeyDown(keyCode, event);
        }
    }
    
    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        logKeyEvent("onKeyUp", keyCode, event);
        
        switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_UP:
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                // Handle volume key release
                return true;
                
            default:
                return super.onKeyUp(keyCode, event);
        }
    }
    
    @Override
    public boolean onKeyLongPress(int keyCode, KeyEvent event) {
        logKeyEvent("onKeyLongPress", keyCode, event);
        
        switch (keyCode) {
            case KeyEvent.KEYCODE_BACK:
                handleBackLongPress();
                return true;
                
            case KeyEvent.KEYCODE_MENU:
                handleMenuLongPress();
                return true;
                
            default:
                return super.onKeyLongPress(keyCode, event);
        }
    }
    
    private void logKeyEvent(String method, int keyCode, KeyEvent event) {
        String keyName = KeyEvent.keyCodeToString(keyCode);
        String logEntry = String.format("%s: %s (code: %d)\n", method, keyName, keyCode);
        
        keyLog.append(logEntry);
        keyEventText.setText(keyLog.toString());
        
        // Scroll to bottom
        keyEventText.post(() -> {
            int scrollAmount = keyEventText.getLayout().getLineTop(keyEventText.getLineCount()) 
                             - keyEventText.getHeight();
            if (scrollAmount > 0) {
                keyEventText.scrollTo(0, scrollAmount);
            }
        });
    }
    
    private void handleVolumeUp() {
        Toast.makeText(this, "Volume Up pressed", Toast.LENGTH_SHORT).show();
    }
    
    private void handleVolumeDown() {
        Toast.makeText(this, "Volume Down pressed", Toast.LENGTH_SHORT).show();
    }
    
    private void handleBackKey() {
        // Custom back button behavior
        new AlertDialog.Builder(this)
            .setTitle("Exit App")
            .setMessage("Are you sure you want to exit?")
            .setPositiveButton("Yes", (dialog, which) -> finish())
            .setNegativeButton("No", null)
            .show();
    }
    
    private void handleMenuKey() {
        // Show custom menu
        PopupMenu popupMenu = new PopupMenu(this, keyEventText);
        popupMenu.getMenuInflater().inflate(R.menu.main_menu, popupMenu.getMenu());
        popupMenu.show();
    }
    
    private void handleSearchKey() {
        // Open search functionality
        Intent searchIntent = new Intent(this, SearchActivity.class);
        startActivity(searchIntent);
    }
    
    private void handleBackLongPress() {
        Toast.makeText(this, "Back key long pressed", Toast.LENGTH_SHORT).show();
    }
    
    private void handleMenuLongPress() {
        Toast.makeText(this, "Menu key long pressed", Toast.LENGTH_SHORT).show();
    }
}
```

### 2. Custom Key Handling in Views
```java
public class CustomKeyView extends View implements View.OnKeyListener {
    
    private Paint paint;
    private String displayText = "Press keys to interact";
    
    public CustomKeyView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    private void init() {
        setFocusable(true);
        setFocusableInTouchMode(true);
        setOnKeyListener(this);
        
        paint = new Paint();
        paint.setColor(Color.BLACK);
        paint.setTextSize(48f);
        paint.setTextAlign(Paint.Align.CENTER);
    }
    
    @Override
    public boolean onKey(View v, int keyCode, KeyEvent event) {
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
            switch (keyCode) {
                case KeyEvent.KEYCODE_DPAD_UP:
                    displayText = "UP arrow pressed";
                    invalidate();
                    return true;
                    
                case KeyEvent.KEYCODE_DPAD_DOWN:
                    displayText = "DOWN arrow pressed";
                    invalidate();
                    return true;
                    
                case KeyEvent.KEYCODE_DPAD_LEFT:
                    displayText = "LEFT arrow pressed";
                    invalidate();
                    return true;
                    
                case KeyEvent.KEYCODE_DPAD_RIGHT:
                    displayText = "RIGHT arrow pressed";
                    invalidate();
                    return true;
                    
                case KeyEvent.KEYCODE_ENTER:
                case KeyEvent.KEYCODE_DPAD_CENTER:
                    displayText = "CENTER/ENTER pressed";
                    invalidate();
                    return true;
                    
                case KeyEvent.KEYCODE_SPACE:
                    displayText = "SPACE pressed";
                    invalidate();
                    return true;
                    
                default:
                    if (event.isPrintingKey()) {
                        char c = (char) event.getUnicodeChar();
                        displayText = "Key pressed: " + c;
                        invalidate();
                        return true;
                    }
                    break;
            }
        }
        
        return false;
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        float centerX = getWidth() / 2f;
        float centerY = getHeight() / 2f;
        
        canvas.drawText(displayText, centerX, centerY, paint);
    }
    
    @Override
    public boolean isFocused() {
        return true; // Always show as focused
    }
}
```

## Text Input Events

### 1. EditText Input Monitoring
```java
public class TextInputActivity extends AppCompatActivity {
    
    private EditText searchField;
    private EditText passwordField;
    private EditText phoneField;
    private TextView inputSummary;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_input);
        
        initializeViews();
        setupTextWatchers();
        setupInputFilters();
    }
    
    private void initializeViews() {
        searchField = findViewById(R.id.searchField);
        passwordField = findViewById(R.id.passwordField);
        phoneField = findViewById(R.id.phoneField);
        inputSummary = findViewById(R.id.inputSummary);
    }
    
    private void setupTextWatchers() {
        // Search field with debouncing
        searchField.addTextChangedListener(new TextWatcher() {
            private final Handler handler = new Handler(Looper.getMainLooper());
            private Runnable searchRunnable;
            
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
                // Called before text is changed
            }
            
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                // Called when text is being changed
                updateInputSummary();
                
                // Cancel previous search
                if (searchRunnable != null) {
                    handler.removeCallbacks(searchRunnable);
                }
                
                // Schedule new search with delay
                searchRunnable = () -> performSearch(s.toString());
                handler.postDelayed(searchRunnable, 300); // 300ms delay
            }
            
            @Override
            public void afterTextChanged(Editable s) {
                // Called after text has been changed
                validateSearchInput(s.toString());
            }
        });
        
        // Password field with strength indicator
        passwordField.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                updatePasswordStrength(s.toString());
                updateInputSummary();
            }
            
            @Override
            public void afterTextChanged(Editable s) {
                validatePassword(s.toString());
            }
        });
        
        // Phone field with formatting
        phoneField.addTextChangedListener(new PhoneNumberFormattingTextWatcher());
        phoneField.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                updateInputSummary();
            }
            
            @Override
            public void afterTextChanged(Editable s) {
                validatePhoneNumber(s.toString());
            }
        });
    }
    
    private void setupInputFilters() {
        // Limit search field to 50 characters
        searchField.setFilters(new InputFilter[]{new InputFilter.LengthFilter(50)});
        
        // Custom filter for phone field (digits, spaces, hyphens, parentheses only)
        InputFilter phoneFilter = (source, start, end, dest, dstart, dend) -> {
            String allowedChars = "0123456789 -()";
            StringBuilder filtered = new StringBuilder();
            
            for (int i = start; i < end; i++) {
                char c = source.charAt(i);
                if (allowedChars.indexOf(c) >= 0) {
                    filtered.append(c);
                }
            }
            
            return filtered.length() == (end - start) ? null : filtered.toString();
        };
        
        phoneField.setFilters(new InputFilter[]{
            phoneFilter,
            new InputFilter.LengthFilter(20)
        });
    }
    
    private void performSearch(String query) {
        if (query.trim().isEmpty()) {
            return;
        }
        
        Log.d("TextInput", "Performing search for: " + query);
        // Implement actual search logic here
        Toast.makeText(this, "Searching for: " + query, Toast.LENGTH_SHORT).show();
    }
    
    private void validateSearchInput(String input) {
        if (input.length() > 0 && input.length() < 3) {
            searchField.setError("Search term must be at least 3 characters");
        } else {
            searchField.setError(null);
        }
    }
    
    private void updatePasswordStrength(String password) {
        ProgressBar strengthBar = findViewById(R.id.passwordStrengthBar);
        TextView strengthText = findViewById(R.id.passwordStrengthText);
        
        int strength = calculatePasswordStrength(password);
        strengthBar.setProgress(strength);
        
        String strengthLabel;
        int color;
        
        if (strength < 25) {
            strengthLabel = "Weak";
            color = Color.RED;
        } else if (strength < 50) {
            strengthLabel = "Fair";
            color = Color.YELLOW;
        } else if (strength < 75) {
            strengthLabel = "Good";
            color = Color.BLUE;
        } else {
            strengthLabel = "Strong";
            color = Color.GREEN;
        }
        
        strengthText.setText(strengthLabel);
        strengthText.setTextColor(color);
    }
    
    private int calculatePasswordStrength(String password) {
        int strength = 0;
        
        if (password.length() >= 8) strength += 25;
        if (password.matches(".*[a-z].*")) strength += 15;
        if (password.matches(".*[A-Z].*")) strength += 15;
        if (password.matches(".*[0-9].*")) strength += 15;
        if (password.matches(".*[!@#$%^&*()].*")) strength += 30;
        
        return Math.min(strength, 100);
    }
    
    private void validatePassword(String password) {
        if (password.length() > 0 && password.length() < 8) {
            passwordField.setError("Password must be at least 8 characters");
        } else {
            passwordField.setError(null);
        }
    }
    
    private void validatePhoneNumber(String phone) {
        String digitsOnly = phone.replaceAll("[^0-9]", "");
        
        if (phone.length() > 0 && digitsOnly.length() < 10) {
            phoneField.setError("Please enter a valid phone number");
        } else {
            phoneField.setError(null);
        }
    }
    
    private void updateInputSummary() {
        String search = searchField.getText().toString();
        String password = passwordField.getText().toString();
        String phone = phoneField.getText().toString();
        
        String summary = String.format(
            "Search: %d chars, Password: %d chars, Phone: %d chars",
            search.length(), password.length(), phone.length()
        );
        
        inputSummary.setText(summary);
    }
}
```

## Gesture Events

### 1. Basic Gesture Detection
```java
public class GestureActivity extends AppCompatActivity implements GestureDetector.OnGestureListener {
    
    private GestureDetector gestureDetector;
    private TextView gestureText;
    private View gestureArea;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gesture);
        
        gestureText = findViewById(R.id.gestureText);
        gestureArea = findViewById(R.id.gestureArea);
        
        gestureDetector = new GestureDetector(this, this);
        
        gestureArea.setOnTouchListener((v, event) -> {
            return gestureDetector.onTouchEvent(event);
        });
    }
    
    @Override
    public boolean onDown(MotionEvent e) {
        gestureText.setText("Gesture: Down");
        return true;
    }
    
    @Override
    public void onShowPress(MotionEvent e) {
        gestureText.setText("Gesture: Show Press");
    }
    
    @Override
    public boolean onSingleTapUp(MotionEvent e) {
        gestureText.setText("Gesture: Single Tap");
        return true;
    }
    
    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
        String direction = getScrollDirection(distanceX, distanceY);
        gestureText.setText("Gesture: Scroll " + direction);
        return true;
    }
    
    @Override
    public void onLongPress(MotionEvent e) {
        gestureText.setText("Gesture: Long Press");
        
        // Vibrate for feedback
        Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        if (vibrator != null && vibrator.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
                vibrator.vibrate(200);
            }
        }
    }
    
    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        String direction = getFlingDirection(e1, e2, velocityX, velocityY);
        gestureText.setText("Gesture: Fling " + direction);
        
        // Handle specific fling directions
        switch (direction) {
            case "Left":
                handleSwipeLeft();
                break;
            case "Right":
                handleSwipeRight();
                break;
            case "Up":
                handleSwipeUp();
                break;
            case "Down":
                handleSwipeDown();
                break;
        }
        
        return true;
    }
    
    private String getScrollDirection(float distanceX, float distanceY) {
        if (Math.abs(distanceX) > Math.abs(distanceY)) {
            return distanceX > 0 ? "Left" : "Right";
        } else {
            return distanceY > 0 ? "Up" : "Down";
        }
    }
    
    private String getFlingDirection(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        float deltaX = e2.getX() - e1.getX();
        float deltaY = e2.getY() - e1.getY();
        
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            return deltaX > 0 ? "Right" : "Left";
        } else {
            return deltaY > 0 ? "Down" : "Up";
        }
    }
    
    private void handleSwipeLeft() {
        Toast.makeText(this, "Swiped Left - Next page", Toast.LENGTH_SHORT).show();
    }
    
    private void handleSwipeRight() {
        Toast.makeText(this, "Swiped Right - Previous page", Toast.LENGTH_SHORT).show();
    }
    
    private void handleSwipeUp() {
        Toast.makeText(this, "Swiped Up - Scroll to top", Toast.LENGTH_SHORT).show();
    }
    
    private void handleSwipeDown() {
        Toast.makeText(this, "Swiped Down - Refresh", Toast.LENGTH_SHORT).show();
    }
}
```

### 2. Scale and Rotation Gestures
```java
public class ScaleRotateView extends View implements ScaleGestureDetector.OnScaleGestureListener {
    
    private ScaleGestureDetector scaleDetector;
    private float scaleFactor = 1.0f;
    private float rotationAngle = 0f;
    private Paint paint;
    private Bitmap bitmap;
    private Matrix matrix;
    
    public ScaleRotateView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }
    
    private void init(Context context) {
        scaleDetector = new ScaleGestureDetector(context, this);
        
        paint = new Paint();
        paint.setAntiAlias(true);
        
        matrix = new Matrix();
        
        // Load a sample bitmap
        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.sample_image);
    }
    
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        scaleDetector.onTouchEvent(event);
        
        // Handle rotation with two fingers
        if (event.getPointerCount() == 2) {
            handleRotation(event);
        }
        
        invalidate();
        return true;
    }
    
    private void handleRotation(MotionEvent event) {
        if (event.getActionMasked() == MotionEvent.ACTION_MOVE) {
            float deltaX = event.getX(1) - event.getX(0);
            float deltaY = event.getY(1) - event.getY(0);
            float currentAngle = (float) Math.toDegrees(Math.atan2(deltaY, deltaX));
            
            // Calculate rotation delta and apply
            rotationAngle = currentAngle;
        }
    }
    
    @Override
    public boolean onScale(ScaleGestureDetector detector) {
        scaleFactor *= detector.getScaleFactor();
        
        // Limit scale factor
        scaleFactor = Math.max(0.1f, Math.min(scaleFactor, 5.0f));
        
        return true;
    }
    
    @Override
    public boolean onScaleBegin(ScaleGestureDetector detector) {
        return true;
    }
    
    @Override
    public void onScaleEnd(ScaleGestureDetector detector) {
        // Scale gesture ended
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        if (bitmap != null) {
            canvas.save();
            
            // Apply transformations
            matrix.reset();
            matrix.postScale(scaleFactor, scaleFactor, getWidth() / 2f, getHeight() / 2f);
            matrix.postRotate(rotationAngle, getWidth() / 2f, getHeight() / 2f);
            
            canvas.setMatrix(matrix);
            
            // Draw bitmap at center
            float x = (getWidth() - bitmap.getWidth()) / 2f;
            float y = (getHeight() - bitmap.getHeight()) / 2f;
            canvas.drawBitmap(bitmap, x, y, paint);
            
            canvas.restore();
        }
        
        // Draw transformation info
        Paint textPaint = new Paint();
        textPaint.setColor(Color.BLACK);
        textPaint.setTextSize(32f);
        canvas.drawText("Scale: " + String.format("%.2f", scaleFactor), 20, 50, textPaint);
        canvas.drawText("Rotation: " + String.format("%.1f°", rotationAngle), 20, 100, textPaint);
    }
}
```

## Custom Event Handling

### 1. Custom Event System
```java
// Custom event interface
public interface CustomEventListener {
    void onCustomEvent(CustomEvent event);
}

// Custom event class
public class CustomEvent {
    public enum Type {
        DATA_LOADED,
        USER_ACTION,
        ERROR_OCCURRED,
        NETWORK_CHANGED
    }
    
    private Type type;
    private Object data;
    private String message;
    private long timestamp;
    
    public CustomEvent(Type type, Object data, String message) {
        this.type = type;
        this.data = data;
        this.message = message;
        this.timestamp = System.currentTimeMillis();
    }
    
    // Getters
    public Type getType() { return type; }
    public Object getData() { return data; }
    public String getMessage() { return message; }
    public long getTimestamp() { return timestamp; }
}

// Event dispatcher
public class EventDispatcher {
    private static EventDispatcher instance;
    private List<CustomEventListener> listeners;
    private Handler mainHandler;
    
    private EventDispatcher() {
        listeners = new ArrayList<>();
        mainHandler = new Handler(Looper.getMainLooper());
    }
    
    public static synchronized EventDispatcher getInstance() {
        if (instance == null) {
            instance = new EventDispatcher();
        }
        return instance;
    }
    
    public void addListener(CustomEventListener listener) {
        if (!listeners.contains(listener)) {
            listeners.add(listener);
        }
    }
    
    public void removeListener(CustomEventListener listener) {
        listeners.remove(listener);
    }
    
    public void dispatchEvent(CustomEvent event) {
        mainHandler.post(() -> {
            for (CustomEventListener listener : new ArrayList<>(listeners)) {
                try {
                    listener.onCustomEvent(event);
                } catch (Exception e) {
                    Log.e("EventDispatcher", "Error dispatching event", e);
                }
            }
        });
    }
    
    public void dispatchEventAsync(CustomEvent event) {
        new Thread(() -> dispatchEvent(event)).start();
    }
}

// Usage in Activity
public class CustomEventActivity extends AppCompatActivity implements CustomEventListener {
    
    private TextView eventLog;
    private StringBuilder logBuilder;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_custom_event);
        
        eventLog = findViewById(R.id.eventLog);
        logBuilder = new StringBuilder();
        
        // Register for events
        EventDispatcher.getInstance().addListener(this);
        
        setupTriggerButtons();
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Unregister to prevent memory leaks
        EventDispatcher.getInstance().removeListener(this);
    }
    
    private void setupTriggerButtons() {
        findViewById(R.id.triggerDataButton).setOnClickListener(v -> {
            CustomEvent event = new CustomEvent(
                CustomEvent.Type.DATA_LOADED,
                "Sample data",
                "Data loaded successfully"
            );
            EventDispatcher.getInstance().dispatchEvent(event);
        });
        
        findViewById(R.id.triggerErrorButton).setOnClickListener(v -> {
            CustomEvent event = new CustomEvent(
                CustomEvent.Type.ERROR_OCCURRED,
                new Exception("Sample error"),
                "An error occurred"
            );
            EventDispatcher.getInstance().dispatchEvent(event);
        });
        
        findViewById(R.id.triggerUserActionButton).setOnClickListener(v -> {
            CustomEvent event = new CustomEvent(
                CustomEvent.Type.USER_ACTION,
                "button_click",
                "User performed an action"
            );
            EventDispatcher.getInstance().dispatchEvent(event);
        });
    }
    
    @Override
    public void onCustomEvent(CustomEvent event) {
        String logEntry = String.format(
            "[%s] %s: %s\n",
            new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date(event.getTimestamp())),
            event.getType().name(),
            event.getMessage()
        );
        
        logBuilder.append(logEntry);
        eventLog.setText(logBuilder.toString());
        
        // Scroll to bottom
        eventLog.post(() -> {
            int scrollAmount = eventLog.getLayout().getLineTop(eventLog.getLineCount()) 
                             - eventLog.getHeight();
            if (scrollAmount > 0) {
                eventLog.scrollTo(0, scrollAmount);
            }
        });
        
        // Handle specific event types
        switch (event.getType()) {
            case DATA_LOADED:
                handleDataLoaded(event);
                break;
            case ERROR_OCCURRED:
                handleError(event);
                break;
            case USER_ACTION:
                handleUserAction(event);
                break;
            case NETWORK_CHANGED:
                handleNetworkChange(event);
                break;
        }
    }
    
    private void handleDataLoaded(CustomEvent event) {
        Toast.makeText(this, "Data loaded: " + event.getData(), Toast.LENGTH_SHORT).show();
    }
    
    private void handleError(CustomEvent event) {
        if (event.getData() instanceof Exception) {
            Exception error = (Exception) event.getData();
            Log.e("CustomEventActivity", "Error occurred", error);
        }
        Toast.makeText(this, "Error: " + event.getMessage(), Toast.LENGTH_LONG).show();
    }
    
    private void handleUserAction(CustomEvent event) {
        Log.d("CustomEventActivity", "User action: " + event.getData());
    }
    
    private void handleNetworkChange(CustomEvent event) {
        // Handle network connectivity changes
        boolean isConnected = (Boolean) event.getData();
        String message = isConnected ? "Connected" : "Disconnected";
        Toast.makeText(this, "Network " + message, Toast.LENGTH_SHORT).show();
    }
}
```

## Best Practices

### 1. Performance Optimization
```java
public class OptimizedEventHandling {
    
    // Use WeakReference to prevent memory leaks
    private WeakReference<Activity> activityRef;
    
    // Throttle rapid events
    private static final int THROTTLE_DELAY = 100; // milliseconds
    private long lastEventTime = 0;
    
    public boolean shouldProcessEvent() {
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastEventTime > THROTTLE_DELAY) {
            lastEventTime = currentTime;
            return true;
        }
        return false;
    }
    
    // Debounce text input
    private Handler debounceHandler = new Handler(Looper.getMainLooper());
    private Runnable debounceRunnable;
    
    public void debounceTextInput(String input, Runnable action) {
        if (debounceRunnable != null) {
            debounceHandler.removeCallbacks(debounceRunnable);
        }
        
        debounceRunnable = action;
        debounceHandler.postDelayed(debounceRunnable, 300);
    }
    
    // Use view pools for heavy operations
    private static final int POOL_SIZE = 10;
    private Pools.SimplePool<View> viewPool = new Pools.SimplePool<>(POOL_SIZE);
    
    public View getPooledView() {
        View view = viewPool.acquire();
        if (view == null) {
            // Create new view if pool is empty
            view = createNewView();
        }
        return view;
    }
    
    public void recycleView(View view) {
        // Reset view state
        resetViewState(view);
        viewPool.release(view);
    }
    
    private View createNewView() {
        // Implementation
        return null;
    }
    
    private void resetViewState(View view) {
        // Reset view to default state
    }
}
```

### 2. Error Handling
```java
public class SafeEventHandling {
    
    public static void safeExecute(Runnable action, String context) {
        try {
            action.run();
        } catch (Exception e) {
            Log.e("SafeEventHandling", "Error in " + context, e);
            
            // Report to crash analytics
            FirebaseCrashlytics.getInstance().recordException(e);
            
            // Show user-friendly message
            // Don't crash the app
        }
    }
    
    public static boolean isActivitySafe(Activity activity) {
        return activity != null && 
               !activity.isFinishing() && 
               !activity.isDestroyed();
    }
    
    public static boolean isFragmentSafe(Fragment fragment) {
        return fragment != null && 
               fragment.isAdded() && 
               !fragment.isDetached() && 
               fragment.getActivity() != null;
    }
    
    // Safe UI updates
    public static void safeUpdateUI(Activity activity, Runnable uiUpdate) {
        if (isActivitySafe(activity)) {
            activity.runOnUiThread(uiUpdate);
        }
    }
    
    public static void safeUpdateUI(Fragment fragment, Runnable uiUpdate) {
        if (isFragmentSafe(fragment) && isActivitySafe(fragment.getActivity())) {
            fragment.requireActivity().runOnUiThread(uiUpdate);
        }
    }
}
```

### 3. Memory Management
```java
public class MemoryEfficientEventHandling {
    
    // Use static inner classes to avoid memory leaks
    private static class StaticClickListener implements View.OnClickListener {
        private WeakReference<Activity> activityRef;
        
        StaticClickListener(Activity activity) {
            this.activityRef = new WeakReference<>(activity);
        }
        
        @Override
        public void onClick(View v) {
            Activity activity = activityRef.get();
            if (activity != null) {
                // Handle click
                handleClick(activity, v);
            }
        }
        
        private void handleClick(Activity activity, View view) {
            // Implementation
        }
    }
    
    // Clean up listeners in lifecycle methods
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        // Remove listeners
        button.setOnClickListener(null);
        editText.removeTextChangedListener(textWatcher);
        
        // Cancel pending operations
        handler.removeCallbacksAndMessages(null);
        
        // Clear references
        adapter = null;
        listener = null;
    }
    
    // Use application context for long-lived operations
    private void setupGlobalListener(Context context) {
        Context appContext = context.getApplicationContext();
        // Use appContext for listeners that outlive activities
    }
}
```

Event handling is fundamental to creating interactive Android applications. Understanding different event types, proper implementation patterns, and best practices ensures your apps are responsive, efficient, and user-friendly.

## Additional Event Handling Concepts (Summary)

### Menu Events
- **Options Menu**: Handle menu item selections in ActionBar/Toolbar
- **Context Menu**: Right-click or long-press contextual menus
- **Popup Menu**: Floating menus attached to views
- **Navigation Drawer**: Side menu navigation events

### Scroll Events
- **ScrollView**: Detect scroll position changes and scroll end
- **RecyclerView**: Handle scroll state changes, load more data
- **NestedScrollView**: Coordinate scrolling between parent and child views
- **ViewPager**: Page change events and scroll detection

### Drag and Drop Events
- **View Dragging**: Move views within layouts
- **Cross-View Drops**: Drop data between different views
- **List Reordering**: Drag to reorder RecyclerView items
- **File Drops**: Handle external file drag operations

### Swipe Events
- **SwipeRefreshLayout**: Pull-to-refresh functionality
- **ItemTouchHelper**: Swipe-to-delete in RecyclerView
- **ViewPager2**: Swipe between pages
- **Custom Swipe**: Implement custom swipe gestures

### System Events
- **Configuration Changes**: Screen rotation, language changes
- **Network Changes**: Connectivity state monitoring
- **Battery Events**: Low battery, charging state
- **Storage Events**: External storage mount/unmount

### Hardware Events
- **Sensor Events**: Accelerometer, gyroscope, proximity
- **Camera Events**: Picture taken, focus changes
- **GPS Events**: Location updates, provider changes
- **Bluetooth Events**: Device connection/disconnection

### Accessibility Events
- **TalkBack**: Screen reader navigation support
- **Focus Events**: Accessibility focus changes
- **Content Description**: Announce content changes
- **Touch Exploration**: Handle accessibility touch modes

### Animation Events
- **Animation Listeners**: Start, end, repeat, cancel callbacks
- **Transition Events**: Shared element transitions
- **Property Animation**: Value animator updates
- **View Animation**: Traditional animation callbacks

### Input Method Events
- **Keyboard Visibility**: Soft keyboard show/hide detection
- **IME Actions**: Done, Next, Search button handling
- **Text Selection**: Handle text selection changes
- **Voice Input**: Speech-to-text recognition

### Application Lifecycle Events
- **Activity Lifecycle**: onCreate, onResume, onPause, etc.
- **Fragment Lifecycle**: Specific to fragment states
- **Service Events**: Background service lifecycle
- **Broadcast Events**: System and custom broadcasts

### Custom Component Events
- **Custom View Events**: Creating custom event types
- **Compound View Events**: Events from custom ViewGroups
- **Widget Events**: App widget interaction events
- **Plugin Events**: Third-party library event integration

### Performance Considerations
- **Event Throttling**: Limit rapid event processing
- **Event Debouncing**: Delay event processing until input stops
- **Memory Management**: Prevent listener memory leaks
- **Background Processing**: Handle events off main thread
- **Event Pooling**: Reuse event objects for efficiency

### Error Handling Strategies
- **Safe Execution**: Wrap event handlers in try-catch
- **Null Checks**: Validate objects before accessing
- **Lifecycle Awareness**: Check component state before updating UI
- **Graceful Degradation**: Provide fallback behavior for failures

This comprehensive coverage ensures your Android applications can handle all types of user interactions and system events effectively.

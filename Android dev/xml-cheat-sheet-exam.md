# Android XML Cheat Sheet & Event Listeners - Written Exam Reference

## Table of Contents
1. [Layout XML Attributes](#layout-xml-attributes)
2. [Common View Attributes](#common-view-attributes)
3. [Layout Types](#layout-types)
4. [Common Views](#common-views)
5. [Event Listeners](#event-listeners)
6. [Resources & Values](#resources--values)
7. [Manifest XML](#manifest-xml)
8. [Menu XML](#menu-xml)
9. [Drawable XML](#drawable-xml)
10. [Animation XML](#animation-xml)

## Layout XML Attributes

### Universal Layout Attributes
```xml
android:layout_width="match_parent|wrap_content|dp"
android:layout_height="match_parent|wrap_content|dp"
android:layout_margin="16dp"
android:layout_marginTop="8dp"
android:layout_marginBottom="8dp"
android:layout_marginLeft="8dp"
android:layout_marginRight="8dp"
android:layout_marginStart="8dp"
android:layout_marginEnd="8dp"
android:padding="16dp"
android:paddingTop="8dp"
android:paddingBottom="8dp"
android:paddingLeft="8dp"
android:paddingRight="8dp"
android:paddingStart="8dp"
android:paddingEnd="8dp"
android:visibility="visible|invisible|gone"
android:background="@color/colorName|@drawable/drawableName|#FFFFFF"
android:id="@+id/viewId"
android:tag="tagValue"
```

### LinearLayout Specific
```xml
android:orientation="vertical|horizontal"
android:layout_weight="1"
android:weightSum="10"
android:gravity="center|start|end|top|bottom|center_vertical|center_horizontal"
android:layout_gravity="center|start|end|top|bottom"
```

### RelativeLayout Specific
```xml
android:layout_alignParentTop="true"
android:layout_alignParentBottom="true"
android:layout_alignParentLeft="true"
android:layout_alignParentRight="true"
android:layout_alignParentStart="true"
android:layout_alignParentEnd="true"
android:layout_centerInParent="true"
android:layout_centerHorizontal="true"
android:layout_centerVertical="true"
android:layout_above="@id/viewId"
android:layout_below="@id/viewId"
android:layout_toLeftOf="@id/viewId"
android:layout_toRightOf="@id/viewId"
android:layout_toStartOf="@id/viewId"
android:layout_toEndOf="@id/viewId"
android:layout_alignTop="@id/viewId"
android:layout_alignBottom="@id/viewId"
android:layout_alignLeft="@id/viewId"
android:layout_alignRight="@id/viewId"
android:layout_alignStart="@id/viewId"
android:layout_alignEnd="@id/viewId"
```

### ConstraintLayout Specific
```xml
app:layout_constraintTop_toTopOf="parent|@id/viewId"
app:layout_constraintBottom_toBottomOf="parent|@id/viewId"
app:layout_constraintLeft_toLeftOf="parent|@id/viewId"
app:layout_constraintRight_toRightOf="parent|@id/viewId"
app:layout_constraintStart_toStartOf="parent|@id/viewId"
app:layout_constraintEnd_toEndOf="parent|@id/viewId"
app:layout_constraintTop_toBottomOf="@id/viewId"
app:layout_constraintBottom_toTopOf="@id/viewId"
app:layout_constraintLeft_toRightOf="@id/viewId"
app:layout_constraintRight_toLeftOf="@id/viewId"
app:layout_constraintHorizontal_bias="0.5"
app:layout_constraintVertical_bias="0.5"
app:layout_constraintWidth_percent="0.5"
app:layout_constraintHeight_percent="0.5"
```

## Common View Attributes

### TextView
```xml
android:text="Hello World"
android:textSize="16sp"
android:textColor="#000000"
android:textStyle="bold|italic|normal"
android:textAlignment="center|textStart|textEnd"
android:gravity="center|start|end"
android:fontFamily="@font/fontName"
android:typeface="normal|sans|serif|monospace"
android:maxLines="2"
android:ellipsize="end|start|middle|marquee"
android:singleLine="true"
android:hint="Enter text"
android:hintTextColor="#CCCCCC"
android:drawableLeft="@drawable/icon"
android:drawableRight="@drawable/icon"
android:drawableTop="@drawable/icon"
android:drawableBottom="@drawable/icon"
android:drawableStart="@drawable/icon"
android:drawableEnd="@drawable/icon"
android:drawablePadding="8dp"
android:autoLink="web|email|phone|map|all"
```

### EditText
```xml
android:inputType="text|textPassword|number|phone|email|date|time"
android:imeOptions="actionDone|actionNext|actionSend|actionGo"
android:maxLength="50"
android:lines="3"
android:minLines="1"
android:scrollbars="vertical"
```

### Button
```xml
android:text="Click Me"
android:onClick="methodName"
android:enabled="true|false"
android:clickable="true|false"
android:background="@drawable/button_selector"
android:backgroundTint="#FF0000"
```

### ImageView
```xml
android:src="@drawable/imageName"
android:scaleType="center|centerCrop|centerInside|fitCenter|fitStart|fitEnd|fitXY|matrix"
android:adjustViewBounds="true"
android:cropToPadding="true"
android:tint="#FF0000"
```

## Layout Types

### LinearLayout
```xml
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:gravity="center"
    android:padding="16dp">
    
    <!-- Child views here -->
    
</LinearLayout>
```

### RelativeLayout
```xml
<RelativeLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    
    <!-- Child views here -->
    
</RelativeLayout>
```

### ConstraintLayout
```xml
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    
    <!-- Child views here -->
    
</androidx.constraintlayout.widget.ConstraintLayout>
```

### FrameLayout
```xml
<FrameLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    
    <!-- Child views here -->
    
</FrameLayout>
```

### GridLayout
```xml
<GridLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:columnCount="2"
    android:rowCount="3">
    
    <!-- Child views here -->
    
</GridLayout>
```

### TableLayout
```xml
<TableLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:stretchColumns="*">
    
    <TableRow>
        <!-- Table cells here -->
    </TableRow>
    
</TableLayout>
```

## Common Views

### TextView
```xml
<TextView
    android:id="@+id/textView"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Hello World"
    android:textSize="18sp"
    android:textColor="#000000" />
```

### EditText
```xml
<EditText
    android:id="@+id/editText"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:hint="Enter text"
    android:inputType="text" />
```

### Button
```xml
<Button
    android:id="@+id/button"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Click Me"
    android:onClick="onButtonClick" />
```

### ImageView
```xml
<ImageView
    android:id="@+id/imageView"
    android:layout_width="100dp"
    android:layout_height="100dp"
    android:src="@drawable/image"
    android:scaleType="centerCrop" />
```

### CheckBox
```xml
<CheckBox
    android:id="@+id/checkBox"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Check me"
    android:checked="false" />
```

### RadioButton & RadioGroup
```xml
<RadioGroup
    android:id="@+id/radioGroup"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:orientation="vertical">
    
    <RadioButton
        android:id="@+id/radioButton1"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Option 1" />
        
    <RadioButton
        android:id="@+id/radioButton2"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Option 2" />
        
</RadioGroup>
```

### Spinner
```xml
<Spinner
    android:id="@+id/spinner"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:entries="@array/spinner_items" />
```

### ListView
```xml
<ListView
    android:id="@+id/listView"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:divider="#CCCCCC"
    android:dividerHeight="1dp" />
```

### RecyclerView
```xml
<androidx.recyclerview.widget.RecyclerView
    android:id="@+id/recyclerView"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:scrollbars="vertical" />
```

### ScrollView
```xml
<ScrollView
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:fillViewport="true">
    
    <!-- Single child view here -->
    
</ScrollView>
```

### ProgressBar
```xml
<ProgressBar
    android:id="@+id/progressBar"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    style="?android:attr/progressBarStyleHorizontal"
    android:max="100"
    android:progress="50" />
```

### SeekBar
```xml
<SeekBar
    android:id="@+id/seekBar"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:max="100"
    android:progress="50" />
```

### Switch
```xml
<Switch
    android:id="@+id/switch1"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Enable feature"
    android:checked="false" />
```

### ToggleButton
```xml
<ToggleButton
    android:id="@+id/toggleButton"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:textOn="ON"
    android:textOff="OFF" />
```

## Event Listeners

### 1. OnClickListener
```java
// In Activity
public void onButtonClick(View view) {
    // Handle click
}

// Programmatically
button.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        // Handle click
    }
});

// Lambda (Java 8+)
button.setOnClickListener(v -> {
    // Handle click
});
```

### 2. OnLongClickListener
```java
view.setOnLongClickListener(new View.OnLongClickListener() {
    @Override
    public boolean onLongClick(View v) {
        // Handle long click
        return true; // Consume the event
    }
});
```

### 3. OnTouchListener
```java
view.setOnTouchListener(new View.OnTouchListener() {
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                // Handle touch down
                break;
            case MotionEvent.ACTION_UP:
                // Handle touch up
                break;
            case MotionEvent.ACTION_MOVE:
                // Handle touch move
                break;
        }
        return true;
    }
});
```

### 4. OnFocusChangeListener
```java
editText.setOnFocusChangeListener(new View.OnFocusChangeListener() {
    @Override
    public void onFocusChange(View v, boolean hasFocus) {
        if (hasFocus) {
            // View gained focus
        } else {
            // View lost focus
        }
    }
});
```

### 5. OnKeyListener
```java
editText.setOnKeyListener(new View.OnKeyListener() {
    @Override
    public boolean onKey(View v, int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_ENTER) {
            // Handle enter key
            return true;
        }
        return false;
    }
});
```

### 6. TextWatcher (for EditText)
```java
editText.addTextChangedListener(new TextWatcher() {
    @Override
    public void beforeTextChanged(CharSequence s, int start, int count, int after) {
        // Before text change
    }

    @Override
    public void onTextChanged(CharSequence s, int start, int before, int count) {
        // During text change
    }

    @Override
    public void afterTextChanged(Editable s) {
        // After text change
    }
});
```

### 7. OnCheckedChangeListener (CheckBox, RadioButton, Switch)
```java
// CheckBox
checkBox.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        // Handle check state change
    }
});

// RadioGroup
radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
    @Override
    public void onCheckedChanged(RadioGroup group, int checkedId) {
        // Handle radio button selection
    }
});
```

### 8. OnItemClickListener (ListView)
```java
listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
    @Override
    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
        // Handle item click
    }
});
```

### 9. OnItemSelectedListener (Spinner)
```java
spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        // Handle item selection
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        // Handle no selection
    }
});
```

### 10. OnSeekBarChangeListener
```java
seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
    @Override
    public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        // Handle progress change
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {
        // Handle start tracking
    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {
        // Handle stop tracking
    }
});
```

## Resources & Values

### strings.xml
```xml
<resources>
    <string name="app_name">My App</string>
    <string name="hello_world">Hello World!</string>
    <string-array name="planets_array">
        <item>Mercury</item>
        <item>Venus</item>
        <item>Earth</item>
    </string-array>
</resources>
```

### colors.xml
```xml
<resources>
    <color name="colorPrimary">#3F51B5</color>
    <color name="colorPrimaryDark">#303F9F</color>
    <color name="colorAccent">#FF4081</color>
    <color name="white">#FFFFFF</color>
    <color name="black">#000000</color>
</resources>
```

### dimens.xml
```xml
<resources>
    <dimen name="activity_horizontal_margin">16dp</dimen>
    <dimen name="activity_vertical_margin">16dp</dimen>
    <dimen name="text_size_large">22sp</dimen>
    <dimen name="text_size_medium">18sp</dimen>
    <dimen name="text_size_small">14sp</dimen>
</resources>
```

### styles.xml
```xml
<resources>
    <style name="AppTheme" parent="Theme.AppCompat.Light.DarkActionBar">
        <item name="colorPrimary">@color/colorPrimary</item>
        <item name="colorPrimaryDark">@color/colorPrimaryDark</item>
        <item name="colorAccent">@color/colorAccent</item>
    </style>
    
    <style name="CustomTextView">
        <item name="android:textSize">16sp</item>
        <item name="android:textColor">#000000</item>
        <item name="android:padding">8dp</item>
    </style>
</resources>
```

## Manifest XML

### AndroidManifest.xml Structure
```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.myapp">

    <!-- Permissions -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />

    <!-- Features -->
    <uses-feature android:name="android.hardware.camera" android:required="true" />

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/AppTheme">

        <!-- Activities -->
        <activity
            android:name=".MainActivity"
            android:label="@string/app_name"
            android:launchMode="singleTop"
            android:screenOrientation="portrait">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

        <!-- Services -->
        <service
            android:name=".MyService"
            android:enabled="true"
            android:exported="false" />

        <!-- Broadcast Receivers -->
        <receiver
            android:name=".MyReceiver"
            android:enabled="true"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.BOOT_COMPLETED" />
            </intent-filter>
        </receiver>

        <!-- Content Providers -->
        <provider
            android:name=".MyContentProvider"
            android:authorities="com.example.myapp.provider"
            android:enabled="true"
            android:exported="false" />

    </application>

</manifest>
```

### Common Permissions
```xml
<!-- Internet -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

<!-- Storage -->
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />

<!-- Location -->
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />

<!-- Camera -->
<uses-permission android:name="android.permission.CAMERA" />

<!-- Phone -->
<uses-permission android:name="android.permission.CALL_PHONE" />
<uses-permission android:name="android.permission.READ_PHONE_STATE" />

<!-- SMS -->
<uses-permission android:name="android.permission.SEND_SMS" />
<uses-permission android:name="android.permission.RECEIVE_SMS" />

<!-- Contacts -->
<uses-permission android:name="android.permission.READ_CONTACTS" />
<uses-permission android:name="android.permission.WRITE_CONTACTS" />
```

## Menu XML

### Options Menu (menu/main.xml)
```xml
<?xml version="1.0" encoding="utf-8"?>
<menu xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto">

    <item
        android:id="@+id/action_settings"
        android:title="Settings"
        android:icon="@drawable/ic_settings"
        app:showAsAction="never" />

    <item
        android:id="@+id/action_search"
        android:title="Search"
        android:icon="@drawable/ic_search"
        app:showAsAction="ifRoom|withText" />

    <item
        android:id="@+id/action_share"
        android:title="Share"
        android:icon="@drawable/ic_share"
        app:showAsAction="ifRoom" />

</menu>
```

### Context Menu
```xml
<?xml version="1.0" encoding="utf-8"?>
<menu xmlns:android="http://schemas.android.com/apk/res/android">

    <item
        android:id="@+id/edit"
        android:title="Edit" />

    <item
        android:id="@+id/delete"
        android:title="Delete" />

    <item
        android:id="@+id/copy"
        android:title="Copy" />

</menu>
```

## Drawable XML

### Shape Drawable
```xml
<?xml version="1.0" encoding="utf-8"?>
<shape xmlns:android="http://schemas.android.com/apk/res/android"
    android:shape="rectangle">

    <solid android:color="#FF0000" />
    <corners android:radius="8dp" />
    <stroke
        android:width="2dp"
        android:color="#000000" />
    <padding
        android:left="10dp"
        android:top="10dp"
        android:right="10dp"
        android:bottom="10dp" />

</shape>
```

### Selector Drawable (State List)
```xml
<?xml version="1.0" encoding="utf-8"?>
<selector xmlns:android="http://schemas.android.com/apk/res/android">

    <item android:state_pressed="true"
        android:drawable="@color/colorPressed" />

    <item android:state_focused="true"
        android:drawable="@color/colorFocused" />

    <item android:state_enabled="false"
        android:drawable="@color/colorDisabled" />

    <item android:drawable="@color/colorNormal" />

</selector>
```

### Layer List Drawable
```xml
<?xml version="1.0" encoding="utf-8"?>
<layer-list xmlns:android="http://schemas.android.com/apk/res/android">

    <item>
        <shape android:shape="rectangle">
            <solid android:color="#FF0000" />
        </shape>
    </item>

    <item android:top="2dp">
        <shape android:shape="rectangle">
            <solid android:color="#00FF00" />
        </shape>
    </item>

</layer-list>
```

## Animation XML

### Translate Animation
```xml
<?xml version="1.0" encoding="utf-8"?>
<translate xmlns:android="http://schemas.android.com/apk/res/android"
    android:fromXDelta="0%"
    android:toXDelta="100%"
    android:fromYDelta="0%"
    android:toYDelta="0%"
    android:duration="1000" />
```

### Alpha Animation
```xml
<?xml version="1.0" encoding="utf-8"?>
<alpha xmlns:android="http://schemas.android.com/apk/res/android"
    android:fromAlpha="0.0"
    android:toAlpha="1.0"
    android:duration="1000" />
```

### Scale Animation
```xml
<?xml version="1.0" encoding="utf-8"?>
<scale xmlns:android="http://schemas.android.com/apk/res/android"
    android:fromXScale="0.0"
    android:toXScale="1.0"
    android:fromYScale="0.0"
    android:toYScale="1.0"
    android:pivotX="50%"
    android:pivotY="50%"
    android:duration="1000" />
```

### Rotate Animation
```xml
<?xml version="1.0" encoding="utf-8"?>
<rotate xmlns:android="http://schemas.android.com/apk/res/android"
    android:fromDegrees="0"
    android:toDegrees="360"
    android:pivotX="50%"
    android:pivotY="50%"
    android:duration="1000"
    android:repeatCount="infinite" />
```

### Animation Set
```xml
<?xml version="1.0" encoding="utf-8"?>
<set xmlns:android="http://schemas.android.com/apk/res/android"
    android:interpolator="@android:anim/accelerate_interpolator">

    <scale
        android:fromXScale="1.0"
        android:toXScale="1.4"
        android:fromYScale="1.0"
        android:toYScale="0.6"
        android:pivotX="50%"
        android:pivotY="50%"
        android:fillAfter="false"
        android:duration="700" />

    <set android:interpolator="@android:anim/decelerate_interpolator">
        <scale
            android:fromXScale="1.4"
            android:toXScale="0.0"
            android:fromYScale="0.6"
            android:toYScale="0.0"
            android:pivotX="50%"
            android:pivotY="50%"
            android:startOffset="700"
            android:duration="400"
            android:fillBefore="false" />

        <rotate
            android:fromDegrees="0"
            android:toDegrees="-45"
            android:toYScale="0.0"
            android:pivotX="50%"
            android:pivotY="50%"
            android:startOffset="700"
            android:duration="400" />
    </set>

</set>
```

## Quick Reference for Exams

### Essential XML Namespaces
```xml
xmlns:android="http://schemas.android.com/apk/res/android"
xmlns:app="http://schemas.android.com/apk/res-auto"
xmlns:tools="http://schemas.android.com/tools"
```

### Common Resource References
```xml
@string/string_name
@color/color_name
@drawable/drawable_name
@dimen/dimen_name
@style/style_name
@id/view_id
@+id/new_view_id
@android:string/ok
@android:color/white
@android:drawable/ic_dialog_alert
```

### Important Measurements
- `dp` (density-independent pixels) - for layout dimensions
- `sp` (scale-independent pixels) - for text sizes
- `px` (pixels) - avoid using
- `in`, `mm`, `pt` - rarely used

### View Visibility States
- `visible` - View is visible and takes space
- `invisible` - View is hidden but takes space
- `gone` - View is hidden and doesn't take space

### Layout Weight Distribution
```xml
<!-- Equal distribution -->
android:layout_weight="1"

<!-- 2:1 ratio -->
android:layout_weight="2"  <!-- First view -->
android:layout_weight="1"  <!-- Second view -->
```

### Common Event Handler Signatures
```java
public void onClick(View view)
public boolean onLongClick(View view)
public boolean onTouch(View view, MotionEvent event)
public void onFocusChange(View view, boolean hasFocus)
public boolean onKey(View view, int keyCode, KeyEvent event)
public void onCheckedChanged(CompoundButton buttonView, boolean isChecked)
public void onItemClick(AdapterView<?> parent, View view, int position, long id)
```

This cheat sheet covers the most important XML attributes, layouts, views, and event listeners you'll need for Android development written exams. Focus on understanding the syntax, common attributes, and event handling patterns.

# Activities and Lifecycle

## Table of Contents
- [What is an Activity](#what-is-an-activity)
- [Activity Lifecycle](#activity-lifecycle)
- [Lifecycle Methods](#lifecycle-methods)
- [Activity States](#activity-states)
- [Creating Activities](#creating-activities)
- [Activity Stack](#activity-stack)
- [Best Practices](#best-practices)
- [Common Scenarios](#common-scenarios)

## What is an Activity

An **Activity** represents a single screen with a user interface. It's one of the fundamental components of an Android application. Each activity is a Java class that extends `Activity` or its subclasses (commonly `AppCompatActivity`).

### Key Characteristics
- Represents a single, focused thing that the user can do
- Has its own lifecycle managed by the Android system
- Can start other activities
- Can be started by other applications (if configured)

### Examples of Activities
- Email app: Compose email activity, inbox activity, read email activity
- Social media app: Login activity, feed activity, profile activity, post creation activity

## Activity Lifecycle

The Android system manages activity lifecycle through a series of method callbacks. Understanding this lifecycle is crucial for creating robust applications.

### Lifecycle Diagram
```
    [Activity Launched]
           ↓
      onCreate()
           ↓
      onStart()
           ↓
      onResume()
           ↓
    [Activity Running]
           ↓
    [Another activity comes into foreground]
           ↓
      onPause()
           ↓
    [Activity no longer visible]
           ↓
      onStop()
           ↓
    [Activity destroyed or returning]
           ↓
    onDestroy() / onRestart()
```

## Lifecycle Methods

### onCreate()
Called when the activity is first created.

```java
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    
    // Initialize UI components
    Button button = findViewById(R.id.button);
    TextView textView = findViewById(R.id.textView);
    
    // Set up listeners
    button.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            textView.setText("Button clicked!");
        }
    });
    
    // Initialize data
    loadUserData();
}
```

**Use cases:**
- Set the content view
- Initialize UI components
- Set up event listeners
- Initialize member variables
- Restore saved state

### onStart()
Called when the activity becomes visible to the user.

```java
@Override
protected void onStart() {
    super.onStart();
    
    // Register broadcast receivers
    registerLocationUpdates();
    
    // Start animations
    startWelcomeAnimation();
}
```

**Use cases:**
- Register broadcast receivers
- Start animations
- Begin location updates

### onResume()
Called when the activity starts interacting with the user.

```java
@Override
protected void onResume() {
    super.onResume();
    
    // Resume camera preview
    if (camera != null) {
        camera.startPreview();
    }
    
    // Resume sensors
    sensorManager.registerListener(this, accelerometer, 
        SensorManager.SENSOR_DELAY_NORMAL);
    
    // Refresh data
    refreshUserData();
}
```

**Use cases:**
- Resume camera/sensor operations
- Resume game play
- Refresh UI with latest data
- Start intensive foreground operations

### onPause()
Called when the system is about to start resuming another activity.

```java
@Override
protected void onPause() {
    super.onPause();
    
    // Pause camera preview
    if (camera != null) {
        camera.stopPreview();
    }
    
    // Unregister sensors
    sensorManager.unregisterListener(this);
    
    // Save user input
    saveUserInput();
}
```

**Use cases:**
- Stop camera/sensor operations
- Pause game play
- Save user input/progress
- Stop CPU-intensive operations

### onStop()
Called when the activity is no longer visible to the user.

```java
@Override
protected void onStop() {
    super.onStop();
    
    // Unregister broadcast receivers
    unregisterLocationUpdates();
    
    // Stop network requests
    cancelNetworkRequests();
    
    // Save application data
    saveApplicationState();
}
```

**Use cases:**
- Stop network requests
- Unregister receivers
- Save application state
- Release resources that aren't needed

### onDestroy()
Called before the activity is destroyed.

```java
@Override
protected void onDestroy() {
    super.onDestroy();
    
    // Release resources
    if (database != null) {
        database.close();
    }
    
    // Cancel background tasks
    if (asyncTask != null) {
        asyncTask.cancel(true);
    }
    
    // Clean up references
    imageView.setImageDrawable(null);
}
```

**Use cases:**
- Final cleanup
- Release database connections
- Cancel background tasks
- Prevent memory leaks

### onRestart()
Called when the activity is restarting after being stopped.

```java
@Override
protected void onRestart() {
    super.onRestart();
    
    // Refresh data that might have changed
    refreshContent();
    
    // Re-initialize components if needed
    reinitializeComponents();
}
```

## Activity States

### Active/Running
- Activity is in the foreground
- Has user focus
- Receiving user input

### Paused
- Activity is partially obscured
- Still visible but doesn't have focus
- Can be killed by system under extreme memory pressure

### Stopped
- Activity is completely hidden
- Retains all state information
- Can be killed by system when memory is needed

### Destroyed
- Activity is removed from memory
- Must be recreated to be used again

## Creating Activities

### Step 1: Create Activity Class
```java
public class SecondActivity extends AppCompatActivity {
    
    public static final String EXTRA_MESSAGE = "com.example.MESSAGE";
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
        
        // Get data from intent
        Intent intent = getIntent();
        String message = intent.getStringExtra(EXTRA_MESSAGE);
        
        // Display the message
        TextView textView = findViewById(R.id.textViewMessage);
        textView.setText(message);
    }
}
```

### Step 2: Create Layout File
```xml
<!-- res/layout/activity_second.xml -->
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:id="@+id/textViewMessage"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Default Message"
        android:textSize="18sp"
        android:gravity="center" />

    <Button
        android:id="@+id/buttonBack"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Go Back"
        android:layout_marginTop="20dp" />

</LinearLayout>
```

### Step 3: Register in Manifest
```xml
<!-- AndroidManifest.xml -->
<application>
    <activity
        android:name=".MainActivity"
        android:exported="true">
        <intent-filter>
            <action android:name="android.intent.action.MAIN" />
            <category android:name="android.intent.category.LAUNCHER" />
        </intent-filter>
    </activity>
    
    <activity
        android:name=".SecondActivity"
        android:exported="false"
        android:parentActivityName=".MainActivity">
        <!-- For API level 15 and below -->
        <meta-data
            android:name="android.support.PARENT_ACTIVITY"
            android:value=".MainActivity" />
    </activity>
</application>
```

### Step 4: Starting the Activity
```java
// In MainActivity
public void openSecondActivity(View view) {
    Intent intent = new Intent(this, SecondActivity.class);
    intent.putExtra(SecondActivity.EXTRA_MESSAGE, "Hello from MainActivity!");
    startActivity(intent);
}
```

## Activity Stack

### Task and Back Stack
- Activities are managed in a **task**
- Each task has a **back stack** of activities
- Last In, First Out (LIFO) order
- Back button pops the current activity

### Launch Modes
Configure in AndroidManifest.xml:

#### standard (default)
```xml
<activity android:name=".MainActivity"
    android:launchMode="standard" />
```
- Creates new instance every time
- Multiple instances can exist

#### singleTop
```xml
<activity android:name=".MainActivity"
    android:launchMode="singleTop" />
```
- Reuses instance if it's at the top of stack
- Calls `onNewIntent()` instead of creating new instance

#### singleTask
```xml
<activity android:name=".MainActivity"
    android:launchMode="singleTask" />
```
- Only one instance exists in the system
- Clears activities above it when resumed

#### singleInstance
```xml
<activity android:name=".MainActivity"
    android:launchMode="singleInstance" />
```
- Only one instance exists in its own task
- No other activities in the same task

### Intent Flags
```java
// Clear back stack
Intent intent = new Intent(this, MainActivity.class);
intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
startActivity(intent);

// Bring to front if exists
intent.setFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
startActivity(intent);
```

## Best Practices

### 1. Save and Restore State
```java
@Override
protected void onSaveInstanceState(Bundle outState) {
    super.onSaveInstanceState(outState);
    
    // Save UI state
    outState.putString("user_input", editText.getText().toString());
    outState.putInt("score", currentScore);
    outState.putBoolean("game_over", isGameOver);
}

@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    
    // Restore state
    if (savedInstanceState != null) {
        String userInput = savedInstanceState.getString("user_input");
        currentScore = savedInstanceState.getInt("score");
        isGameOver = savedInstanceState.getBoolean("game_over");
        
        // Restore UI
        editText.setText(userInput);
        scoreTextView.setText(String.valueOf(currentScore));
    }
}
```

### 2. Handle Configuration Changes
```java
// In AndroidManifest.xml
<activity android:name=".MainActivity"
    android:configChanges="orientation|screenSize|keyboardHidden" />

// In Activity
@Override
public void onConfigurationChanged(Configuration newConfig) {
    super.onConfigurationChanged(newConfig);
    
    if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE) {
        // Handle landscape orientation
    } else if (newConfig.orientation == Configuration.ORIENTATION_PORTRAIT) {
        // Handle portrait orientation
    }
}
```

### 3. Proper Resource Management
```java
public class MainActivity extends AppCompatActivity {
    private BroadcastReceiver networkReceiver;
    private LocationManager locationManager;
    
    @Override
    protected void onResume() {
        super.onResume();
        
        // Register receivers
        networkReceiver = new NetworkReceiver();
        registerReceiver(networkReceiver, new IntentFilter(ConnectivityManager.CONNECTIVITY_ACTION));
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        
        // Unregister receivers
        if (networkReceiver != null) {
            unregisterReceiver(networkReceiver);
            networkReceiver = null;
        }
    }
}
```

### 4. Memory Management
```java
@Override
protected void onDestroy() {
    super.onDestroy();
    
    // Cancel background tasks
    if (downloadTask != null && !downloadTask.isCancelled()) {
        downloadTask.cancel(true);
    }
    
    // Clear image caches
    if (imageView != null) {
        Drawable drawable = imageView.getDrawable();
        if (drawable instanceof BitmapDrawable) {
            BitmapDrawable bitmapDrawable = (BitmapDrawable) drawable;
            Bitmap bitmap = bitmapDrawable.getBitmap();
            if (bitmap != null && !bitmap.isRecycled()) {
                bitmap.recycle();
            }
        }
        imageView.setImageDrawable(null);
    }
}
```

## Common Scenarios

### 1. Login Flow
```java
public class LoginActivity extends AppCompatActivity {
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Check if already logged in
        if (isUserLoggedIn()) {
            startMainActivity();
            finish(); // Close login activity
            return;
        }
        
        setContentView(R.layout.activity_login);
        setupLoginForm();
    }
    
    private void onLoginSuccess() {
        // Save login state
        saveLoginState();
        
        // Start main activity and clear back stack
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        finish();
    }
}
```

### 2. Splash Screen
```java
public class SplashActivity extends AppCompatActivity {
    private static final int SPLASH_DELAY = 2000; // 2 seconds
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);
        
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent intent = new Intent(SplashActivity.this, MainActivity.class);
                startActivity(intent);
                finish();
            }
        }, SPLASH_DELAY);
    }
}
```

### 3. Activity Result Handling
```java
public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CODE_SETTINGS = 100;
    
    public void openSettings(View view) {
        Intent intent = new Intent(this, SettingsActivity.class);
        startActivityForResult(intent, REQUEST_CODE_SETTINGS);
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (requestCode == REQUEST_CODE_SETTINGS) {
            if (resultCode == RESULT_OK && data != null) {
                boolean settingsChanged = data.getBooleanExtra("settings_changed", false);
                if (settingsChanged) {
                    // Refresh UI based on new settings
                    refreshUI();
                }
            }
        }
    }
}
```

### 4. Handling Back Press
```java
@Override
public void onBackPressed() {
    if (webView.canGoBack()) {
        webView.goBack();
    } else if (fragmentManager.getBackStackEntryCount() > 0) {
        fragmentManager.popBackStack();
    } else {
        // Show exit confirmation
        showExitConfirmation();
    }
}

private void showExitConfirmation() {
    new AlertDialog.Builder(this)
            .setTitle("Exit App")
            .setMessage("Are you sure you want to exit?")
            .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    finish();
                }
            })
            .setNegativeButton("No", null)
            .show();
}
```

Understanding the activity lifecycle is fundamental to creating responsive and stable Android applications. Proper lifecycle management ensures your app behaves correctly in various scenarios and provides a smooth user experience.

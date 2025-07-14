# Intents and Navigation

## Table of Contents
- [What are Intents](#what-are-intents)
- [Types of Intents](#types-of-intents)
- [Explicit Intents](#explicit-intents)
- [Implicit Intents](#implicit-intents)
- [Intent Filters](#intent-filters)
- [Passing Data](#passing-data)
- [Activity Results](#activity-results)
- [Common Intent Actions](#common-intent-actions)
- [Navigation Patterns](#navigation-patterns)
- [Best Practices](#best-practices)

## What are Intents

An **Intent** is a messaging object used to request an action from another app component. Intents facilitate communication between activities, services, and broadcast receivers.

### Key Uses of Intents
1. Starting activities
2. Starting services
3. Delivering broadcasts
4. Communicating between app components

### Intent Structure
```java
Intent intent = new Intent();
intent.setAction(Intent.ACTION_VIEW);           // Action
intent.setData(Uri.parse("https://google.com")); // Data
intent.putExtra("key", "value");                // Extras
intent.setType("text/plain");                   // MIME type
intent.addCategory(Intent.CATEGORY_DEFAULT);    // Category
```

## Types of Intents

### 1. Explicit Intents
Specify the exact component to start.

```java
// Starting a specific activity
Intent intent = new Intent(MainActivity.this, SecondActivity.class);
startActivity(intent);

// Starting a service
Intent serviceIntent = new Intent(this, MyService.class);
startService(serviceIntent);
```

### 2. Implicit Intents
Specify the type of action to perform, letting the system choose the component.

```java
// Open a web page
Intent intent = new Intent(Intent.ACTION_VIEW);
intent.setData(Uri.parse("https://www.google.com"));
startActivity(intent);

// Send an email
Intent emailIntent = new Intent(Intent.ACTION_SEND);
emailIntent.setType("text/plain");
emailIntent.putExtra(Intent.EXTRA_EMAIL, new String[]{"recipient@example.com"});
emailIntent.putExtra(Intent.EXTRA_SUBJECT, "Subject");
emailIntent.putExtra(Intent.EXTRA_TEXT, "Email body");
startActivity(Intent.createChooser(emailIntent, "Send email"));
```

## Explicit Intents

### Basic Activity Navigation
```java
public class MainActivity extends AppCompatActivity {
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        Button button = findViewById(R.id.buttonNext);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                navigateToSecondActivity();
            }
        });
    }
    
    private void navigateToSecondActivity() {
        Intent intent = new Intent(MainActivity.this, SecondActivity.class);
        startActivity(intent);
    }
}
```

### Finishing Activities
```java
// Close current activity and return to previous
finish();

// Close and clear all activities above target
Intent intent = new Intent(this, MainActivity.class);
intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
startActivity(intent);

// Start new task and clear all previous activities
Intent intent = new Intent(this, MainActivity.class);
intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
startActivity(intent);
finish();
```

### Intent Flags
```java
// Common flags
Intent intent = new Intent(this, MainActivity.class);

// Bring existing activity to front instead of creating new
intent.setFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);

// Clear all activities above target
intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);

// Start in new task
intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);

// Single top - don't create if already at top
intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);

// No history - remove from back stack when left
intent.setFlags(Intent.FLAG_ACTIVITY_NO_HISTORY);

startActivity(intent);
```

## Implicit Intents

### Common Implicit Intent Examples

#### 1. Open Website
```java
public void openWebsite(String url) {
    Intent intent = new Intent(Intent.ACTION_VIEW);
    intent.setData(Uri.parse(url));
    
    // Check if there's an app that can handle this intent
    if (intent.resolveActivity(getPackageManager()) != null) {
        startActivity(intent);
    } else {
        Toast.makeText(this, "No app found to open this link", Toast.LENGTH_SHORT).show();
    }
}
```

#### 2. Make Phone Call
```java
public void makePhoneCall(String phoneNumber) {
    Intent intent = new Intent(Intent.ACTION_CALL);
    intent.setData(Uri.parse("tel:" + phoneNumber));
    
    // Check permission
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE) 
            == PackageManager.PERMISSION_GRANTED) {
        startActivity(intent);
    } else {
        // Request permission
        ActivityCompat.requestPermissions(this, 
            new String[]{Manifest.permission.CALL_PHONE}, 
            REQUEST_CALL_PERMISSION);
    }
}

// For dial without permission
public void dialPhoneNumber(String phoneNumber) {
    Intent intent = new Intent(Intent.ACTION_DIAL);
    intent.setData(Uri.parse("tel:" + phoneNumber));
    startActivity(intent);
}
```

#### 3. Send SMS
```java
public void sendSMS(String phoneNumber, String message) {
    Intent intent = new Intent(Intent.ACTION_SENDTO);
    intent.setData(Uri.parse("smsto:" + phoneNumber));
    intent.putExtra("sms_body", message);
    
    if (intent.resolveActivity(getPackageManager()) != null) {
        startActivity(intent);
    }
}
```

#### 4. Send Email
```java
public void sendEmail(String[] recipients, String subject, String body) {
    Intent intent = new Intent(Intent.ACTION_SEND);
    intent.setType("text/plain");
    intent.putExtra(Intent.EXTRA_EMAIL, recipients);
    intent.putExtra(Intent.EXTRA_SUBJECT, subject);
    intent.putExtra(Intent.EXTRA_TEXT, body);
    
    if (intent.resolveActivity(getPackageManager()) != null) {
        startActivity(Intent.createChooser(intent, "Send email"));
    }
}
```

#### 5. Share Content
```java
public void shareText(String text) {
    Intent intent = new Intent(Intent.ACTION_SEND);
    intent.setType("text/plain");
    intent.putExtra(Intent.EXTRA_TEXT, text);
    
    Intent chooser = Intent.createChooser(intent, "Share via");
    if (intent.resolveActivity(getPackageManager()) != null) {
        startActivity(chooser);
    }
}

public void shareImage(Uri imageUri) {
    Intent intent = new Intent(Intent.ACTION_SEND);
    intent.setType("image/*");
    intent.putExtra(Intent.EXTRA_STREAM, imageUri);
    intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
    
    startActivity(Intent.createChooser(intent, "Share image"));
}
```

#### 6. Open Map Location
```java
public void openMap(double latitude, double longitude, String label) {
    String uri = String.format("geo:%f,%f?q=%f,%f(%s)", 
        latitude, longitude, latitude, longitude, label);
    Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(uri));
    intent.setPackage("com.google.android.apps.maps"); // Optional: force Google Maps
    
    if (intent.resolveActivity(getPackageManager()) != null) {
        startActivity(intent);
    } else {
        // Fallback to browser
        String browserUri = String.format("https://maps.google.com/maps?q=%f,%f(%s)", 
            latitude, longitude, label);
        Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(browserUri));
        startActivity(browserIntent);
    }
}
```

#### 7. Capture Photo
```java
private static final int REQUEST_IMAGE_CAPTURE = 1;

public void capturePhoto() {
    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
    if (intent.resolveActivity(getPackageManager()) != null) {
        startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
    }
}

@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    
    if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
        Bundle extras = data.getExtras();
        Bitmap imageBitmap = (Bitmap) extras.get("data");
        ImageView imageView = findViewById(R.id.imageView);
        imageView.setImageBitmap(imageBitmap);
    }
}
```

#### 8. Pick Contact
```java
private static final int REQUEST_PICK_CONTACT = 2;

public void pickContact() {
    Intent intent = new Intent(Intent.ACTION_PICK, ContactsContract.Contacts.CONTENT_URI);
    startActivityForResult(intent, REQUEST_PICK_CONTACT);
}

@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    
    if (requestCode == REQUEST_PICK_CONTACT && resultCode == RESULT_OK) {
        Uri contactUri = data.getData();
        // Process contact data
    }
}
```

## Intent Filters

Define what intents your activity can respond to in `AndroidManifest.xml`.

### Basic Intent Filter
```xml
<activity android:name=".ViewActivity">
    <intent-filter>
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.DEFAULT" />
        <data android:mimeType="text/plain" />
    </intent-filter>
</activity>
```

### Multiple Actions
```xml
<activity android:name=".ShareActivity">
    <intent-filter>
        <action android:name="android.intent.action.SEND" />
        <action android:name="android.intent.action.SEND_MULTIPLE" />
        <category android:name="android.intent.category.DEFAULT" />
        <data android:mimeType="text/*" />
        <data android:mimeType="image/*" />
    </intent-filter>
</activity>
```

### Custom Actions
```xml
<activity android:name=".CustomActivity">
    <intent-filter>
        <action android:name="com.example.myapp.CUSTOM_ACTION" />
        <category android:name="android.intent.category.DEFAULT" />
    </intent-filter>
</activity>
```

```java
// Trigger custom action
Intent intent = new Intent("com.example.myapp.CUSTOM_ACTION");
startActivity(intent);
```

### Data Schemes
```xml
<activity android:name=".WebViewActivity">
    <intent-filter>
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.DEFAULT" />
        <category android:name="android.intent.category.BROWSABLE" />
        <data android:scheme="https"
              android:host="example.com" />
    </intent-filter>
</activity>
```

## Passing Data

### Primitive Data Types
```java
// Sending activity
Intent intent = new Intent(this, SecondActivity.class);
intent.putExtra("string_key", "Hello World");
intent.putExtra("int_key", 42);
intent.putExtra("boolean_key", true);
intent.putExtra("double_key", 3.14);
intent.putExtra("long_key", 123456789L);
startActivity(intent);

// Receiving activity
Intent intent = getIntent();
String stringValue = intent.getStringExtra("string_key");
int intValue = intent.getIntExtra("int_key", 0); // Default value
boolean boolValue = intent.getBooleanExtra("boolean_key", false);
double doubleValue = intent.getDoubleExtra("double_key", 0.0);
long longValue = intent.getLongExtra("long_key", 0L);
```

### Arrays and Collections
```java
// Sending
String[] stringArray = {"item1", "item2", "item3"};
ArrayList<String> stringList = new ArrayList<>(Arrays.asList(stringArray));
int[] intArray = {1, 2, 3, 4, 5};

Intent intent = new Intent(this, SecondActivity.class);
intent.putExtra("string_array", stringArray);
intent.putStringArrayListExtra("string_list", stringList);
intent.putExtra("int_array", intArray);
startActivity(intent);

// Receiving
String[] stringArray = intent.getStringArrayExtra("string_array");
ArrayList<String> stringList = intent.getStringArrayListExtra("string_list");
int[] intArray = intent.getIntArrayExtra("int_array");
```

### Serializable Objects
```java
// User class implementing Serializable
public class User implements Serializable {
    private String name;
    private int age;
    private String email;
    
    // Constructors, getters, setters
    public User(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
    }
    
    // Getters and setters...
}

// Sending
User user = new User("John Doe", 30, "john@example.com");
Intent intent = new Intent(this, SecondActivity.class);
intent.putExtra("user_object", user);
startActivity(intent);

// Receiving
User user = (User) intent.getSerializableExtra("user_object");
```

### Parcelable Objects (Recommended)
```java
// User class implementing Parcelable
public class User implements Parcelable {
    private String name;
    private int age;
    private String email;
    
    public User(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
    }
    
    protected User(Parcel in) {
        name = in.readString();
        age = in.readInt();
        email = in.readString();
    }
    
    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(name);
        dest.writeInt(age);
        dest.writeString(email);
    }
    
    @Override
    public int describeContents() {
        return 0;
    }
    
    public static final Creator<User> CREATOR = new Creator<User>() {
        @Override
        public User createFromParcel(Parcel in) {
            return new User(in);
        }
        
        @Override
        public User[] newArray(int size) {
            return new User[size];
        }
    };
    
    // Getters and setters...
}

// Sending
User user = new User("John Doe", 30, "john@example.com");
Intent intent = new Intent(this, SecondActivity.class);
intent.putExtra("user_object", user);
startActivity(intent);

// Receiving
User user = intent.getParcelableExtra("user_object");
```

### Bundle for Multiple Data
```java
// Sending
Bundle bundle = new Bundle();
bundle.putString("name", "John");
bundle.putInt("age", 30);
bundle.putStringArray("hobbies", new String[]{"reading", "swimming"});

Intent intent = new Intent(this, SecondActivity.class);
intent.putExtras(bundle);
startActivity(intent);

// Receiving
Bundle bundle = intent.getExtras();
if (bundle != null) {
    String name = bundle.getString("name");
    int age = bundle.getInt("age");
    String[] hobbies = bundle.getStringArray("hobbies");
}
```

## Activity Results

### Traditional approach (Deprecated in API 30+)
```java
private static final int REQUEST_CODE_SETTINGS = 100;

// Starting activity for result
public void openSettings() {
    Intent intent = new Intent(this, SettingsActivity.class);
    startActivityForResult(intent, REQUEST_CODE_SETTINGS);
}

// Handling result
@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    
    if (requestCode == REQUEST_CODE_SETTINGS) {
        if (resultCode == RESULT_OK && data != null) {
            boolean settingsChanged = data.getBooleanExtra("settings_changed", false);
            String newTheme = data.getStringExtra("theme");
            
            if (settingsChanged) {
                applyNewSettings(newTheme);
            }
        } else if (resultCode == RESULT_CANCELED) {
            // User canceled
        }
    }
}

// In SettingsActivity - returning result
public void saveSettings() {
    Intent resultIntent = new Intent();
    resultIntent.putExtra("settings_changed", true);
    resultIntent.putExtra("theme", selectedTheme);
    setResult(RESULT_OK, resultIntent);
    finish();
}
```

### Modern Approach - Activity Result API
```java
public class MainActivity extends AppCompatActivity {
    
    // Register for activity result
    private ActivityResultLauncher<Intent> settingsLauncher = 
        registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == RESULT_OK) {
                        Intent data = result.getData();
                        if (data != null) {
                            boolean settingsChanged = data.getBooleanExtra("settings_changed", false);
                            String newTheme = data.getStringExtra("theme");
                            
                            if (settingsChanged) {
                                applyNewSettings(newTheme);
                            }
                        }
                    }
                }
            });
    
    // Launch activity
    public void openSettings() {
        Intent intent = new Intent(this, SettingsActivity.class);
        settingsLauncher.launch(intent);
    }
}
```

### Permission Request with Activity Result API
```java
private ActivityResultLauncher<String> requestPermissionLauncher =
    registerForActivityResult(new ActivityResultContracts.RequestPermission(),
        new ActivityResultCallback<Boolean>() {
            @Override
            public void onActivityResult(Boolean isGranted) {
                if (isGranted) {
                    // Permission granted
                    makePhoneCall();
                } else {
                    // Permission denied
                    Toast.makeText(MainActivity.this, "Permission denied", Toast.LENGTH_SHORT).show();
                }
            }
        });

// Request permission
public void requestCallPermission() {
    requestPermissionLauncher.launch(Manifest.permission.CALL_PHONE);
}
```

## Common Intent Actions

### Standard Actions
```java
// View/display data
Intent.ACTION_VIEW

// Edit data
Intent.ACTION_EDIT

// Send data to someone else
Intent.ACTION_SEND

// Send data to multiple recipients
Intent.ACTION_SEND_MULTIPLE

// Pick an item from data
Intent.ACTION_PICK

// Get content from provider
Intent.ACTION_GET_CONTENT

// Dial a number
Intent.ACTION_DIAL

// Call a number
Intent.ACTION_CALL

// Send SMS to someone
Intent.ACTION_SENDTO

// Capture image/video
MediaStore.ACTION_IMAGE_CAPTURE
MediaStore.ACTION_VIDEO_CAPTURE

// Install package
Intent.ACTION_PACKAGE_INSTALL

// Uninstall package
Intent.ACTION_DELETE

// Open settings
Settings.ACTION_SETTINGS
Settings.ACTION_WIFI_SETTINGS
Settings.ACTION_APPLICATION_DETAILS_SETTINGS
```

### System Settings
```java
// Open app-specific settings
public void openAppSettings() {
    Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
    Uri uri = Uri.fromParts("package", getPackageName(), null);
    intent.setData(uri);
    startActivity(intent);
}

// Open Wi-Fi settings
public void openWifiSettings() {
    Intent intent = new Intent(Settings.ACTION_WIFI_SETTINGS);
    startActivity(intent);
}

// Open location settings
public void openLocationSettings() {
    Intent intent = new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS);
    startActivity(intent);
}
```

## Navigation Patterns

### 1. Up Navigation
```xml
<!-- In AndroidManifest.xml -->
<activity
    android:name=".DetailActivity"
    android:parentActivityName=".MainActivity">
    <meta-data
        android:name="android.support.PARENT_ACTIVITY"
        android:value=".MainActivity" />
</activity>
```

```java
// In DetailActivity
@Override
public boolean onOptionsItemSelected(MenuItem item) {
    switch (item.getItemId()) {
        case android.R.id.home:
            // Navigate up
            Intent upIntent = NavUtils.getParentActivityIntent(this);
            if (NavUtils.shouldUpRecreateTask(this, upIntent)) {
                // Create new task
                TaskStackBuilder.create(this)
                    .addNextIntentWithParentStack(upIntent)
                    .startActivities();
            } else {
                // Navigate up to existing task
                NavUtils.navigateUpTo(this, upIntent);
            }
            return true;
    }
    return super.onOptionsItemSelected(item);
}
```

### 2. Deep Linking
```xml
<!-- In AndroidManifest.xml -->
<activity android:name=".ProductDetailActivity">
    <intent-filter android:autoVerify="true">
        <action android:name="android.intent.action.VIEW" />
        <category android:name="android.intent.category.DEFAULT" />
        <category android:name="android.intent.category.BROWSABLE" />
        <data android:scheme="https"
              android:host="myapp.com"
              android:pathPrefix="/product" />
    </intent-filter>
</activity>
```

```java
// Handle deep link in ProductDetailActivity
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_product_detail);
    
    Intent intent = getIntent();
    String action = intent.getAction();
    Uri data = intent.getData();
    
    if (Intent.ACTION_VIEW.equals(action) && data != null) {
        String productId = data.getLastPathSegment();
        loadProduct(productId);
    }
}
```

### 3. Task Management
```java
// Clear task and start new
public void logout() {
    Intent intent = new Intent(this, LoginActivity.class);
    intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
    startActivity(intent);
    finish();
}

// Bring existing activity to front
public void goToMain() {
    Intent intent = new Intent(this, MainActivity.class);
    intent.setFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
    startActivity(intent);
}

// Single top behavior
public void goToCurrentActivityTop() {
    Intent intent = new Intent(this, getClass());
    intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
    startActivity(intent);
}

// Handle new intent when activity is single top
@Override
protected void onNewIntent(Intent intent) {
    super.onNewIntent(intent);
    // Handle new intent
    processIntent(intent);
}
```

## Best Practices

### 1. Validate Intent Recipients
```java
public void sendEmail(String email, String subject, String body) {
    Intent intent = new Intent(Intent.ACTION_SEND);
    intent.setType("text/plain");
    intent.putExtra(Intent.EXTRA_EMAIL, new String[]{email});
    intent.putExtra(Intent.EXTRA_SUBJECT, subject);
    intent.putExtra(Intent.EXTRA_TEXT, body);
    
    // Always check if there's an app to handle the intent
    if (intent.resolveActivity(getPackageManager()) != null) {
        startActivity(intent);
    } else {
        Toast.makeText(this, "No email app found", Toast.LENGTH_SHORT).show();
    }
}
```

### 2. Use Constants for Keys
```java
public class IntentKeys {
    public static final String EXTRA_USER_ID = "user_id";
    public static final String EXTRA_PRODUCT_ID = "product_id";
    public static final String EXTRA_MESSAGE = "message";
}

// Usage
Intent intent = new Intent(this, DetailActivity.class);
intent.putExtra(IntentKeys.EXTRA_USER_ID, userId);
startActivity(intent);
```

### 3. Handle Null Intent Data
```java
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_detail);
    
    Intent intent = getIntent();
    if (intent != null) {
        String userId = intent.getStringExtra(IntentKeys.EXTRA_USER_ID);
        if (userId != null) {
            loadUserData(userId);
        } else {
            // Handle missing data
            showError("User ID not provided");
            finish();
        }
    }
}
```

### 4. Use Appropriate Launch Modes
```xml
<!-- For main/launcher activity -->
<activity
    android:name=".MainActivity"
    android:launchMode="singleTop">
    
<!-- For login activity -->
<activity
    android:name=".LoginActivity"
    android:launchMode="singleTask"
    android:clearTaskOnLaunch="true">
```

### 5. Secure Intent Handling
```java
// Verify the intent is from trusted source for sensitive actions
if (intent.getAction() != null && intent.getAction().equals("com.example.SENSITIVE_ACTION")) {
    String callingPackage = getCallingPackage();
    if ("com.trusted.app".equals(callingPackage)) {
        // Process trusted intent
    } else {
        // Reject untrusted intent
        return;
    }
}
```

Intents are the foundation of Android app navigation and inter-component communication. Understanding how to use them effectively is crucial for building well-connected Android applications.

# Android Project Structure

## Table of Contents
- [Overview](#overview)
- [App Module Structure](#app-module-structure)
- [Source Sets](#source-sets)
- [Resources Directory](#resources-directory)
- [Gradle Files](#gradle-files)
- [Generated Files](#generated-files)
- [Best Practices](#best-practices)

## Overview

Understanding the Android project structure is crucial for efficient development. An Android project consists of modules, with the main module being the `app` module.

```
MyProject/
├── app/                    # Main application module
├── gradle/                 # Gradle wrapper files
├── build.gradle           # Project-level build file
├── gradle.properties      # Project-wide Gradle settings
├── settings.gradle        # Project settings
└── local.properties       # Local SDK path (not in version control)
```

## App Module Structure

### Complete Directory Structure
```
app/
├── build/                 # Generated build files (not in version control)
├── libs/                  # Local library files (.jar, .aar)
├── src/
│   ├── main/             # Main source set
│   │   ├── java/         # Java source code
│   │   │   └── com/example/myapp/
│   │   │       ├── MainActivity.java
│   │   │       ├── models/
│   │   │       ├── adapters/
│   │   │       ├── fragments/
│   │   │       └── utils/
│   │   ├── res/          # Resources directory
│   │   │   ├── drawable/ # Images and drawable resources
│   │   │   ├── layout/   # XML layout files
│   │   │   ├── values/   # Values (strings, colors, styles)
│   │   │   ├── menu/     # Menu XML files
│   │   │   ├── raw/      # Raw asset files
│   │   │   └── xml/      # Arbitrary XML files
│   │   └── AndroidManifest.xml
│   ├── test/             # Unit tests
│   └── androidTest/      # Instrumented tests
├── build.gradle          # Module-level build file
└── proguard-rules.pro    # ProGuard configuration
```

## Source Sets

### Main Source Set (`src/main/`)
Contains the primary application code and resources.

#### Java Directory (`src/main/java/`)
```
java/
└── com/
    └── example/
        └── myapp/
            ├── MainActivity.java          # Main activity
            ├── activities/               # Other activities
            │   ├── LoginActivity.java
            │   └── ProfileActivity.java
            ├── fragments/               # Fragment classes
            │   ├── HomeFragment.java
            │   └── SettingsFragment.java
            ├── adapters/               # RecyclerView adapters
            │   └── UserAdapter.java
            ├── models/                 # Data models/POJOs
            │   ├── User.java
            │   └── Product.java
            ├── utils/                  # Utility classes
            │   ├── NetworkUtils.java
            │   └── DateUtils.java
            ├── database/              # Database related classes
            │   ├── AppDatabase.java
            │   └── UserDao.java
            └── services/              # Background services
                └── DataSyncService.java
```

### Test Source Sets
#### Unit Tests (`src/test/`)
```java
// Example: src/test/java/com/example/myapp/UtilsTest.java
@RunWith(JUnit4.class)
public class UtilsTest {
    @Test
    public void testDateFormat() {
        // Test logic here
    }
}
```

#### Instrumented Tests (`src/androidTest/`)
```java
// Example: src/androidTest/java/com/example/myapp/MainActivityTest.java
@RunWith(AndroidJUnit4.class)
public class MainActivityTest {
    @Test
    public void testButtonClick() {
        // UI test logic here
    }
}
```

## Resources Directory

### Layout Files (`res/layout/`)
XML files defining UI structure:
```
layout/
├── activity_main.xml      # Main activity layout
├── activity_login.xml     # Login activity layout
├── fragment_home.xml      # Home fragment layout
├── item_user.xml         # RecyclerView item layout
└── dialog_custom.xml     # Custom dialog layout
```

### Drawable Resources (`res/drawable/`)
Images and drawable definitions:
```
drawable/
├── ic_launcher.png        # App icon
├── button_background.xml  # Custom button background
├── gradient_background.xml # Gradient drawable
└── user_placeholder.png   # Placeholder image
```

#### Vector Drawable Example
```xml
<!-- res/drawable/ic_heart.xml -->
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp"
    android:height="24dp"
    android:viewportWidth="24"
    android:viewportHeight="24">
    <path
        android:fillColor="#FF0000"
        android:pathData="M12,21.35l-1.45,-1.32C5.4,15.36 2,12.28 2,8.5 2,5.42 4.42,3 7.5,3c1.74,0 3.41,0.81 4.5,2.09C13.09,3.81 14.76,3 16.5,3 19.58,3 22,5.42 22,8.5c0,3.78 -3.4,6.86 -8.55,11.54L12,21.35z"/>
</vector>
```

### Values Directory (`res/values/`)

#### Strings (`values/strings.xml`)
```xml
<resources>
    <string name="app_name">My App</string>
    <string name="login">Login</string>
    <string name="welcome_message">Welcome, %1$s!</string>
    <string name="error_network">Network connection error</string>
</resources>
```

#### Colors (`values/colors.xml`)
```xml
<resources>
    <color name="colorPrimary">#6200EE</color>
    <color name="colorPrimaryDark">#3700B3</color>
    <color name="colorAccent">#03DAC5</color>
    <color name="white">#FFFFFF</color>
    <color name="black">#000000</color>
</resources>
```

#### Styles (`values/styles.xml`)
```xml
<resources>
    <style name="AppTheme" parent="Theme.AppCompat.Light.DarkActionBar">
        <item name="colorPrimary">@color/colorPrimary</item>
        <item name="colorPrimaryDark">@color/colorPrimaryDark</item>
        <item name="colorAccent">@color/colorAccent</item>
    </style>
    
    <style name="CustomButton">
        <item name="android:layout_width">match_parent</item>
        <item name="android:layout_height">wrap_content</item>
        <item name="android:background">@drawable/button_background</item>
    </style>
</resources>
```

#### Dimensions (`values/dimens.xml`)
```xml
<resources>
    <dimen name="margin_small">8dp</dimen>
    <dimen name="margin_medium">16dp</dimen>
    <dimen name="margin_large">24dp</dimen>
    <dimen name="text_size_small">12sp</dimen>
    <dimen name="text_size_medium">16sp</dimen>
    <dimen name="text_size_large">20sp</dimen>
</resources>
```

### Menu Files (`res/menu/`)
```xml
<!-- res/menu/main_menu.xml -->
<menu xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto">
    
    <item
        android:id="@+id/action_search"
        android:icon="@drawable/ic_search"
        android:title="Search"
        app:showAsAction="ifRoom" />
        
    <item
        android:id="@+id/action_settings"
        android:title="Settings"
        app:showAsAction="never" />
</menu>
```

## Gradle Files

### Project-level build.gradle
```gradle
// Top-level build file
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:7.4.2'
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

task clean(type: Delete) {
    delete rootProject.buildDir
}
```

### Module-level build.gradle (app/build.gradle)
```gradle
plugins {
    id 'com.android.application'
}

android {
    namespace 'com.example.myapp'
    compileSdk 34

    defaultConfig {
        applicationId "com.example.myapp"
        minSdk 21
        targetSdk 34
        versionCode 1
        versionName "1.0"
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
}

dependencies {
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.9.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
}
```

## Generated Files

### R.java Class
Automatically generated class containing resource IDs:
```java
public final class R {
    public static final class id {
        public static final int button_login = 0x7f080001;
        public static final int text_username = 0x7f080002;
    }
    
    public static final class layout {
        public static final int activity_main = 0x7f0b0001;
    }
    
    public static final class string {
        public static final int app_name = 0x7f0f0001;
    }
}
```

### BuildConfig.java
Contains build configuration constants:
```java
public final class BuildConfig {
    public static final boolean DEBUG = Boolean.parseBoolean("true");
    public static final String APPLICATION_ID = "com.example.myapp";
    public static final String BUILD_TYPE = "debug";
    public static final int VERSION_CODE = 1;
    public static final String VERSION_NAME = "1.0";
}
```

## Best Practices

### 1. Package Organization
```
com.example.myapp/
├── activities/        # All activity classes
├── fragments/         # All fragment classes  
├── adapters/         # RecyclerView adapters
├── models/           # Data models
├── database/         # Database related classes
├── network/          # Network/API classes
├── utils/            # Utility classes
├── services/         # Background services
└── broadcast/        # Broadcast receivers
```

### 2. Resource Naming Conventions

#### Layout Files
- Activities: `activity_<name>.xml`
- Fragments: `fragment_<name>.xml`
- List items: `item_<name>.xml`
- Dialogs: `dialog_<name>.xml`

#### Drawable Files
- Icons: `ic_<name>_<color>_<size>.xml`
- Backgrounds: `bg_<description>.xml`
- Selectors: `selector_<name>.xml`

#### String Resources
- Use descriptive names: `error_network_connection`
- Group related strings: `login_title`, `login_button`, `login_error`

### 3. Code Organization
```java
public class MainActivity extends AppCompatActivity {
    // Constants
    private static final String TAG = "MainActivity";
    private static final int REQUEST_CODE = 100;
    
    // UI components
    private Button loginButton;
    private EditText usernameEditText;
    
    // Data
    private User currentUser;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Implementation
    }
    
    // Private methods
    private void initViews() {
        // Initialize UI components
    }
    
    private void setupListeners() {
        // Set up event listeners
    }
}
```

### 4. Resource Management
- Use `dimens.xml` for consistent spacing
- Define colors in `colors.xml` (avoid hardcoded colors)
- Extract strings to `strings.xml` for localization
- Use appropriate drawable densities (mdpi, hdpi, xhdpi, xxhdpi)

### 5. Version Control
#### .gitignore Example
```
# Built application files
*.apk
*.aar

# Files for the ART/Dalvik VM
*.dex

# Java class files
*.class

# Generated files
bin/
gen/
out/
build/

# Local configuration file (sdk path, etc)
local.properties

# Android Studio files
.idea/
*.iml

# Gradle files
.gradle/
```

## Alternative Project Structures

### Feature-based Structure
```
com.example.myapp/
├── login/
│   ├── LoginActivity.java
│   ├── LoginFragment.java
│   └── LoginPresenter.java
├── profile/
│   ├── ProfileActivity.java
│   └── ProfileFragment.java
└── common/
    ├── utils/
    ├── models/
    └── database/
```

### MVP Architecture Structure
```
com.example.myapp/
├── data/
│   ├── models/
│   ├── repositories/
│   └── database/
├── ui/
│   ├── login/
│   │   ├── LoginActivity.java
│   │   ├── LoginPresenter.java
│   │   └── LoginContract.java
│   └── main/
│       ├── MainActivity.java
│       ├── MainPresenter.java
│       └── MainContract.java
└── utils/
```

Understanding and following a well-organized project structure is essential for maintainable and scalable Android applications.

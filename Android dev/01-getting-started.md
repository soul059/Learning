# Getting Started with Android Development

## Table of Contents
- [System Requirements](#system-requirements)
- [Installing Android Studio](#installing-android-studio)
- [Setting up the SDK](#setting-up-the-sdk)
- [Creating Your First Project](#creating-your-first-project)
- [Running the App](#running-the-app)
- [Understanding the Basics](#understanding-the-basics)

## System Requirements

### Minimum Requirements
- **Windows**: Windows 8/10/11 (64-bit)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk Space**: 4 GB minimum (500 MB for IDE + 3.5 GB for SDK)
- **Screen Resolution**: 1280 x 800 minimum

### Recommended
- **RAM**: 16 GB or more
- **CPU**: Intel i5 or AMD equivalent
- **SSD**: For faster build times

## Installing Android Studio

1. **Download Android Studio**
   - Visit [developer.android.com/studio](https://developer.android.com/studio)
   - Download the latest stable version
   - Choose the version for your operating system

2. **Installation Process**
   - Run the downloaded installer
   - Follow the setup wizard
   - Choose "Standard" installation type
   - Accept license agreements

3. **First Launch**
   - Launch Android Studio
   - Complete the setup wizard
   - Download necessary SDK components

## Setting up the SDK

### SDK Manager
```
Tools → SDK Manager
```

### Essential SDK Components
1. **Android SDK Platform-Tools**
2. **Android SDK Build-Tools** (latest version)
3. **Android API Levels**:
   - Latest stable version (API 34+)
   - Target API for your app
   - Minimum API you want to support

### SDK Tools
- **Android Emulator**
- **Intel x86 Emulator Accelerator (HAXM)**
- **Google Play Services**
- **Google Repository**

## Creating Your First Project

### Step 1: Start New Project
1. Open Android Studio
2. Click "Start a new Android Studio project"
3. Choose "Empty Activity" template

### Step 2: Configure Project
```
Application name: MyFirstApp
Package name: com.example.myfirstapp
Save location: Choose your preferred directory
Language: Java
Minimum API level: API 21 (Android 5.0)
```

### Step 3: Project Creation
- Android Studio will create the project structure
- Gradle will sync and download dependencies
- Wait for indexing to complete

## Running the App

### Using Android Virtual Device (AVD)
1. **Create AVD**:
   ```
   Tools → AVD Manager → Create Virtual Device
   ```

2. **Choose Device**:
   - Select device definition (e.g., Pixel 4)
   - Choose system image (latest Android version)
   - Configure AVD settings

3. **Run the App**:
   - Click "Run" button (green triangle)
   - Select your AVD
   - Wait for emulator to boot
   - App will install and launch

### Using Physical Device
1. **Enable Developer Options**:
   ```
   Settings → About Phone → Tap "Build Number" 7 times
   ```

2. **Enable USB Debugging**:
   ```
   Settings → Developer Options → USB Debugging
   ```

3. **Connect Device**:
   - Connect via USB
   - Allow USB debugging when prompted
   - Select device in Android Studio and run

## Understanding the Basics

### Project Files Structure
```
app/
├── src/
│   ├── main/
│   │   ├── java/com/example/myfirstapp/
│   │   │   └── MainActivity.java
│   │   ├── res/
│   │   │   ├── layout/
│   │   │   │   └── activity_main.xml
│   │   │   ├── values/
│   │   │   │   ├── strings.xml
│   │   │   │   └── colors.xml
│   │   │   └── mipmap/
│   │   └── AndroidManifest.xml
│   └── test/
└── build.gradle (Module: app)
```

### Key Concepts

#### 1. Activity
- Single screen with user interface
- Entry point for user interaction
- Extends `AppCompatActivity` class

#### 2. Layout
- XML files defining UI structure
- Located in `res/layout/` directory
- Describes arrangement of views

#### 3. Manifest
- `AndroidManifest.xml` declares app components
- Defines permissions, activities, services
- App metadata and configuration

#### 4. Resources
- External files like strings, colors, images
- Located in `res/` directory
- Accessed via R class

### Hello World Example

#### MainActivity.java
```java
package com.example.myfirstapp;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        TextView textView = findViewById(R.id.textView);
        textView.setText("Hello, Android World!");
    }
}
```

#### activity_main.xml
```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:gravity="center">

    <TextView
        android:id="@+id/textView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!"
        android:textSize="24sp" />

</LinearLayout>
```

## Common Issues and Solutions

### 1. Gradle Sync Issues
```
File → Invalidate Caches and Restart
```

### 2. Emulator Performance
- Enable hardware acceleration
- Allocate sufficient RAM to AVD
- Use x86 system images

### 3. SDK License Issues
```
Tools → SDK Manager → SDK Tools → Accept licenses
```

### 4. Build Errors
- Check Gradle version compatibility
- Verify SDK versions
- Clean and rebuild project

## Next Steps
- Explore the [Project Structure](./02-project-structure.md)
- Learn about [Activities and Lifecycle](./03-activities-lifecycle.md)
- Practice with different layout types

## Useful Keyboard Shortcuts
- **Ctrl + Shift + F10**: Run current file
- **Ctrl + F9**: Build project
- **Ctrl + Shift + A**: Find action
- **Alt + Enter**: Quick fix
- **Ctrl + Space**: Code completion

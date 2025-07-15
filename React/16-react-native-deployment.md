# React Native Building and Deployment

## Table of Contents
- [Build Environment Setup](#build-environment-setup)
- [iOS Deployment](#ios-deployment)
- [Android Deployment](#android-deployment)
- [Expo Managed Workflow](#expo-managed-workflow)
- [Over-the-Air Updates](#over-the-air-updates)
- [CI/CD Pipeline](#cicd-pipeline)
- [App Store Optimization](#app-store-optimization)
- [Release Management](#release-management)

## Build Environment Setup

### Development Environment
```bash
# React Native CLI setup
npm install -g react-native-cli

# iOS development (macOS only)
# Install Xcode from App Store
sudo xcode-select --install
sudo gem install cocoapods

# Android development
# Download Android Studio
# Set ANDROID_HOME environment variable
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/tools
export PATH=$PATH:$ANDROID_HOME/tools/bin
export PATH=$PATH:$ANDROID_HOME/platform-tools

# Java Development Kit
brew install openjdk@11
sudo ln -sfn /usr/local/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk

# Verify installation
npx react-native doctor
```

### Build Configuration
```javascript
// metro.config.js
const { getDefaultConfig } = require('@react-native/metro-config');

const config = getDefaultConfig(__dirname);

// Custom configuration for builds
config.transformer = {
  ...config.transformer,
  minifierConfig: {
    // Optimize for production
    mangle: {
      keep_fnames: true,
    },
    output: {
      ascii_only: true,
      quote_style: 3,
      wrap_iife: true,
    },
    sourceMap: {
      includeSources: false,
    },
    toplevel: false,
    warnings: false,
    ie8: false,
    keep_fnames: true,
  },
};

// Bundle splitting
config.serializer = {
  ...config.serializer,
  createModuleIdFactory: () => (path) => {
    // Generate consistent module IDs
    return require('crypto')
      .createHash('sha1')
      .update(path)
      .digest('hex')
      .substr(0, 8);
  },
};

module.exports = config;
```

### Environment Configuration
```javascript
// config/index.js
const isDev = __DEV__;
const isStaging = process.env.NODE_ENV === 'staging';
const isProduction = process.env.NODE_ENV === 'production';

const config = {
  development: {
    API_URL: 'http://localhost:3000/api',
    ANALYTICS_ENABLED: false,
    LOGGING_ENABLED: true,
    CRASH_REPORTING: false,
  },
  staging: {
    API_URL: 'https://staging-api.example.com/api',
    ANALYTICS_ENABLED: true,
    LOGGING_ENABLED: true,
    CRASH_REPORTING: true,
  },
  production: {
    API_URL: 'https://api.example.com/api',
    ANALYTICS_ENABLED: true,
    LOGGING_ENABLED: false,
    CRASH_REPORTING: true,
  },
};

const getConfig = () => {
  if (isDev) return config.development;
  if (isStaging) return config.staging;
  return config.production;
};

export default getConfig();

// Usage
import config from './config';

const apiService = {
  baseURL: config.API_URL,
  // ... other configurations
};
```

## iOS Deployment

### Xcode Project Configuration
```javascript
// ios/YourApp/Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>Your App Name</string>
  <key>CFBundleExecutable</key>
  <string>$(EXECUTABLE_NAME)</string>
  <key>CFBundleIdentifier</key>
  <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>$(PRODUCT_NAME)</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  
  <!-- Permissions -->
  <key>NSCameraUsageDescription</key>
  <string>This app needs access to camera to take photos</string>
  <key>NSPhotoLibraryUsageDescription</key>
  <string>This app needs access to photo library to select images</string>
  <key>NSLocationWhenInUseUsageDescription</key>
  <string>This app needs access to location to provide location-based features</string>
  
  <!-- App Transport Security -->
  <key>NSAppTransportSecurity</key>
  <dict>
    <key>NSExceptionDomains</key>
    <dict>
      <key>localhost</key>
      <dict>
        <key>NSExceptionAllowsInsecureHTTPLoads</key>
        <true/>
      </dict>
    </dict>
  </dict>
  
  <!-- URL Schemes -->
  <key>CFBundleURLTypes</key>
  <array>
    <dict>
      <key>CFBundleURLName</key>
      <string>yourapp</string>
      <key>CFBundleURLSchemes</key>
      <array>
        <string>yourapp</string>
      </array>
    </dict>
  </array>
</dict>
</plist>
```

### iOS Build Scripts
```bash
#!/bin/bash
# scripts/ios-build.sh

set -e

echo "🏗️ Building iOS app..."

# Clean build folder
rm -rf ios/build

# Install pods
cd ios
pod install --repo-update
cd ..

# Build for different configurations
case "$1" in
  "debug")
    echo "Building Debug configuration..."
    npx react-native run-ios --configuration Debug
    ;;
  "release")
    echo "Building Release configuration..."
    npx react-native run-ios --configuration Release
    ;;
  "archive")
    echo "Creating Archive for App Store..."
    cd ios
    xcodebuild -workspace YourApp.xcworkspace \
               -scheme YourApp \
               -configuration Release \
               -destination generic/platform=iOS \
               -archivePath build/YourApp.xcarchive \
               archive
    
    # Export IPA
    xcodebuild -exportArchive \
               -archivePath build/YourApp.xcarchive \
               -exportPath build \
               -exportOptionsPlist ExportOptions.plist
    cd ..
    ;;
  *)
    echo "Usage: $0 {debug|release|archive}"
    exit 1
    ;;
esac

echo "✅ iOS build completed!"
```

### ExportOptions.plist
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key>
  <string>app-store</string>
  <key>teamID</key>
  <string>YOUR_TEAM_ID</string>
  <key>uploadBitcode</key>
  <false/>
  <key>uploadSymbols</key>
  <true/>
  <key>compileBitcode</key>
  <false/>
  <key>stripSwiftSymbols</key>
  <true/>
  <key>thinning</key>
  <string>&lt;none&gt;</string>
</dict>
</plist>
```

### Fastlane iOS Configuration
```ruby
# ios/fastlane/Fastfile
default_platform(:ios)

platform :ios do
  desc "Push a new beta build to TestFlight"
  lane :beta do
    increment_build_number(xcodeproj: "YourApp.xcodeproj")
    build_app(workspace: "YourApp.xcworkspace", scheme: "YourApp")
    upload_to_testflight
  end

  desc "Push a new release build to the App Store"
  lane :release do
    increment_build_number(xcodeproj: "YourApp.xcodeproj")
    build_app(workspace: "YourApp.xcworkspace", scheme: "YourApp")
    upload_to_app_store
  end

  desc "Create screenshots"
  lane :screenshots do
    capture_screenshots
    upload_to_app_store(skip_binary_upload: true, skip_metadata: true)
  end
end
```

## Android Deployment

### Gradle Configuration
```gradle
// android/app/build.gradle
apply plugin: "com.android.application"
apply plugin: "com.facebook.react"

android {
    compileSdkVersion rootProject.ext.compileSdkVersion

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }

    defaultConfig {
        applicationId "com.yourcompany.yourapp"
        minSdkVersion rootProject.ext.minSdkVersion
        targetSdkVersion rootProject.ext.targetSdkVersion
        versionCode 1
        versionName "1.0"
        
        // Enable multidex for large apps
        multiDexEnabled true
        
        // Proguard
        proguardFiles getDefaultProguardFile("proguard-android.txt"), "proguard-rules.pro"
    }

    signingConfigs {
        debug {
            storeFile file('debug.keystore')
            storePassword 'android'
            keyAlias 'androiddebugkey'
            keyPassword 'android'
        }
        release {
            if (project.hasProperty('MYAPP_UPLOAD_STORE_FILE')) {
                storeFile file(MYAPP_UPLOAD_STORE_FILE)
                storePassword MYAPP_UPLOAD_STORE_PASSWORD
                keyAlias MYAPP_UPLOAD_KEY_ALIAS
                keyPassword MYAPP_UPLOAD_KEY_PASSWORD
            }
        }
    }

    buildTypes {
        debug {
            signingConfig signingConfigs.debug
            minifyEnabled false
            proguardFiles getDefaultProguardFile("proguard-android.txt"), "proguard-rules.pro"
        }
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile("proguard-android.txt"), "proguard-rules.pro"
            
            // Enable code shrinking
            shrinkResources true
            
            // Split APKs by ABI
            splits {
                abi {
                    reset()
                    enable true
                    universalApk false
                    include "armeabi-v7a", "x86", "arm64-v8a", "x86_64"
                }
            }
        }
    }

    // Bundle configuration
    bundle {
        density {
            enableSplit true
        }
        abi {
            enableSplit true
        }
        language {
            enableSplit false
        }
    }
}

dependencies {
    implementation fileTree(dir: "libs", include: ["*.jar"])
    implementation "com.facebook.react:react-native:+"
    implementation "androidx.multidex:multidex:2.0.1"
    
    if (enableHermes) {
        def hermesPath = "../../node_modules/hermes-engine/android/"
        debugImplementation files(hermesPath + "hermes-debug.aar")
        releaseImplementation files(hermesPath + "hermes-release.aar")
    } else {
        implementation jscFlavor
    }
}
```

### Android Build Scripts
```bash
#!/bin/bash
# scripts/android-build.sh

set -e

echo "🤖 Building Android app..."

# Clean build
cd android
./gradlew clean
cd ..

# Build based on configuration
case "$1" in
  "debug")
    echo "Building Debug APK..."
    cd android
    ./gradlew assembleDebug
    cd ..
    echo "✅ Debug APK created at: android/app/build/outputs/apk/debug/app-debug.apk"
    ;;
  "release")
    echo "Building Release APK..."
    cd android
    ./gradlew assembleRelease
    cd ..
    echo "✅ Release APK created at: android/app/build/outputs/apk/release/"
    ;;
  "bundle")
    echo "Building Android App Bundle (AAB)..."
    cd android
    ./gradlew bundleRelease
    cd ..
    echo "✅ App Bundle created at: android/app/build/outputs/bundle/release/app-release.aab"
    ;;
  *)
    echo "Usage: $0 {debug|release|bundle}"
    exit 1
    ;;
esac
```

### ProGuard Configuration
```pro
# android/app/proguard-rules.pro

# React Native
-keep class com.facebook.react.** { *; }
-keep class com.facebook.hermes.reactexecutor.** { *; }
-keep class com.facebook.jni.** { *; }

# Application classes
-keep class com.yourcompany.yourapp.** { *; }

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-keep class sun.misc.Unsafe { *; }
-keep class com.google.gson.stream.** { *; }

# Keep native method names
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep custom views
-keep public class * extends android.view.View {
    public <init>(android.content.Context);
    public <init>(android.content.Context, android.util.AttributeSet);
    public <init>(android.content.Context, android.util.AttributeSet, int);
    public void set*(...);
}

# Firebase
-keep class com.google.firebase.** { *; }
-keep class com.google.android.gms.** { *; }
```

### Fastlane Android Configuration
```ruby
# android/fastlane/Fastfile
default_platform(:android)

platform :android do
  desc "Runs all the tests"
  lane :test do
    gradle(task: "test")
  end

  desc "Submit a new Beta Build to Play Console"
  lane :beta do
    gradle(task: "clean bundleRelease")
    upload_to_play_store(
      track: 'internal',
      release_status: 'draft',
      aab: '../build/outputs/bundle/release/app-release.aab'
    )
  end

  desc "Deploy a new version to the Google Play"
  lane :deploy do
    gradle(task: "clean bundleRelease")
    upload_to_play_store(
      track: 'production',
      release_status: 'draft',
      aab: '../build/outputs/bundle/release/app-release.aab'
    )
  end
end
```

## Expo Managed Workflow

### EAS Build Configuration
```json
// eas.json
{
  "cli": {
    "version": ">= 3.0.0"
  },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "ios": {
        "resourceClass": "m1-medium"
      },
      "android": {
        "buildType": "developmentClient"
      }
    },
    "preview": {
      "distribution": "internal",
      "ios": {
        "simulator": true,
        "resourceClass": "m1-medium"
      },
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "ios": {
        "resourceClass": "m1-medium"
      },
      "android": {
        "buildType": "aab"
      }
    }
  },
  "submit": {
    "production": {
      "ios": {
        "appleId": "your-apple-id@example.com",
        "ascAppId": "1234567890",
        "appleTeamId": "ABCD123456"
      },
      "android": {
        "serviceAccountKeyPath": "../path/to/api-key.json",
        "track": "production"
      }
    }
  }
}
```

### EAS Build Commands
```bash
# Install EAS CLI
npm install -g eas-cli

# Login to Expo
eas login

# Configure project
eas build:configure

# Build for development
eas build --profile development --platform ios
eas build --profile development --platform android

# Build for preview
eas build --profile preview --platform all

# Build for production
eas build --profile production --platform all

# Submit to stores
eas submit --platform ios
eas submit --platform android

# Check build status
eas build:list

# View build logs
eas build:view [BUILD_ID]
```

### App Configuration for EAS
```javascript
// app.config.js
export default ({ config }) => {
  const isProduction = process.env.NODE_ENV === 'production';
  
  return {
    ...config,
    name: isProduction ? "Your App" : "Your App (Dev)",
    slug: "your-app",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    userInterfaceStyle: "light",
    splash: {
      image: "./assets/splash.png",
      resizeMode: "contain",
      backgroundColor: "#ffffff"
    },
    updates: {
      fallbackToCacheTimeout: 0,
      url: "https://u.expo.dev/your-project-id"
    },
    assetBundlePatterns: ["**/*"],
    ios: {
      supportsTablet: true,
      bundleIdentifier: isProduction 
        ? "com.yourcompany.yourapp" 
        : "com.yourcompany.yourapp.dev",
      buildNumber: process.env.BUILD_NUMBER || "1"
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#FFFFFF"
      },
      package: isProduction 
        ? "com.yourcompany.yourapp" 
        : "com.yourcompany.yourapp.dev",
      versionCode: parseInt(process.env.BUILD_NUMBER || "1", 10)
    },
    web: {
      favicon: "./assets/favicon.png"
    },
    extra: {
      apiUrl: process.env.API_URL,
      environment: process.env.NODE_ENV,
      eas: {
        projectId: "your-project-id"
      }
    }
  };
};
```

## Over-the-Air Updates

### EAS Update Setup
```bash
# Install EAS Update
npx expo install expo-updates

# Configure EAS Update
eas update:configure

# Publish update
eas update --branch production --message "Bug fixes and improvements"

# Publish to specific branch
eas update --branch staging --message "New features for testing"

# View update history
eas update:list

# Delete update
eas update:delete [UPDATE_ID]
```

### Update Configuration
```javascript
// App.js - Update handling
import { useEffect } from 'react';
import * as Updates from 'expo-updates';
import { Alert } from 'react-native';

function App() {
  useEffect(() => {
    checkForUpdates();
  }, []);

  const checkForUpdates = async () => {
    if (!__DEV__) {
      try {
        const update = await Updates.checkForUpdateAsync();
        
        if (update.isAvailable) {
          Alert.alert(
            'Update Available',
            'A new version is available. Would you like to update now?',
            [
              { text: 'Later', style: 'cancel' },
              { text: 'Update', onPress: downloadAndRestart },
            ]
          );
        }
      } catch (error) {
        console.error('Error checking for updates:', error);
      }
    }
  };

  const downloadAndRestart = async () => {
    try {
      await Updates.fetchUpdateAsync();
      await Updates.reloadAsync();
    } catch (error) {
      console.error('Error downloading update:', error);
      Alert.alert('Update Failed', 'Failed to download update. Please try again later.');
    }
  };

  // Rest of your app
  return <YourAppContent />;
}

// Update service
class UpdateService {
  static async checkForUpdate() {
    if (__DEV__ || !Updates.isEnabled) {
      return { isAvailable: false };
    }

    try {
      const update = await Updates.checkForUpdateAsync();
      return update;
    } catch (error) {
      console.error('Update check failed:', error);
      return { isAvailable: false };
    }
  }

  static async downloadUpdate(onProgress) {
    try {
      const downloadResult = await Updates.fetchUpdateAsync();
      
      if (downloadResult.isNew) {
        return { success: true, manifest: downloadResult.manifest };
      }
      
      return { success: false, reason: 'No new update' };
    } catch (error) {
      console.error('Update download failed:', error);
      return { success: false, error: error.message };
    }
  }

  static async applyUpdate() {
    try {
      await Updates.reloadAsync();
    } catch (error) {
      console.error('Update apply failed:', error);
      throw error;
    }
  }

  static getUpdateInfo() {
    return {
      updateId: Updates.updateId,
      createdAt: Updates.createdAt,
      runtimeVersion: Updates.runtimeVersion,
      isEmbeddedLaunch: Updates.isEmbeddedLaunch,
    };
  }
}
```

### CodePush Alternative
```javascript
// CodePush integration (for bare React Native)
import codePush from 'react-native-code-push';

// CodePush options
const codePushOptions = {
  checkFrequency: codePush.CheckFrequency.ON_APP_RESUME,
  installMode: codePush.InstallMode.ON_NEXT_RESUME,
  minimumBackgroundDuration: 60000, // 1 minute
  updateDialog: {
    title: 'Update Available',
    description: 'An update is available. Would you like to install it?',
    mandatoryUpdateMessage: 'An update is required to continue.',
    mandatoryContinueButtonLabel: 'Install',
    optionalIgnoreButtonLabel: 'Later',
    optionalInstallButtonLabel: 'Install',
  },
};

// Wrap your app component
const App = () => {
  return <YourAppContent />;
};

export default codePush(codePushOptions)(App);

// Manual update check
const checkForUpdate = () => {
  codePush.sync({
    updateDialog: true,
    installMode: codePush.InstallMode.IMMEDIATE,
  });
};
```

## CI/CD Pipeline

### GitHub Actions Configuration
```yaml
# .github/workflows/build-and-deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test
      
      - name: Run linting
        run: npm run lint
      
      - name: Type check
        run: npm run type-check

  build-ios:
    needs: test
    runs-on: macos-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Setup Expo
        uses: expo/expo-github-action@v8
        with:
          expo-version: latest
          token: ${{ secrets.EXPO_TOKEN }}
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build iOS
        run: eas build --platform ios --non-interactive
        env:
          EXPO_TOKEN: ${{ secrets.EXPO_TOKEN }}

  build-android:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Setup Expo
        uses: expo/expo-github-action@v8
        with:
          expo-version: latest
          token: ${{ secrets.EXPO_TOKEN }}
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build Android
        run: eas build --platform android --non-interactive
        env:
          EXPO_TOKEN: ${{ secrets.EXPO_TOKEN }}

  deploy:
    needs: [build-ios, build-android]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Expo
        uses: expo/expo-github-action@v8
        with:
          expo-version: latest
          token: ${{ secrets.EXPO_TOKEN }}
      
      - name: Publish Update
        run: eas update --branch production --message "Automated deployment from main"
        env:
          EXPO_TOKEN: ${{ secrets.EXPO_TOKEN }}
```

### GitLab CI Configuration
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  NODE_VERSION: "18"

test:
  stage: test
  image: node:$NODE_VERSION
  cache:
    paths:
      - node_modules/
  script:
    - npm ci
    - npm run test
    - npm run lint
    - npm run type-check
  artifacts:
    reports:
      junit: junit.xml
      coverage: coverage/cobertura-coverage.xml

build:ios:
  stage: build
  image: node:$NODE_VERSION
  only:
    - main
  before_script:
    - npm install -g eas-cli
  script:
    - npm ci
    - eas build --platform ios --non-interactive
  variables:
    EXPO_TOKEN: $EXPO_TOKEN

build:android:
  stage: build
  image: node:$NODE_VERSION
  only:
    - main
  before_script:
    - npm install -g eas-cli
  script:
    - npm ci
    - eas build --platform android --non-interactive
  variables:
    EXPO_TOKEN: $EXPO_TOKEN

deploy:
  stage: deploy
  image: node:$NODE_VERSION
  only:
    - main
  dependencies:
    - build:ios
    - build:android
  before_script:
    - npm install -g eas-cli
  script:
    - eas update --branch production --message "Deployment from GitLab CI"
  variables:
    EXPO_TOKEN: $EXPO_TOKEN
```

## App Store Optimization

### App Store Connect Configuration
```javascript
// App Store metadata configuration
const appStoreMetadata = {
  app_name: "Your Awesome App",
  subtitle: "The best app for productivity",
  description: `
    Transform your daily workflow with our innovative productivity app.
    
    KEY FEATURES:
    • Intuitive task management
    • Real-time collaboration
    • Advanced analytics
    • Offline synchronization
    
    Perfect for individuals and teams looking to boost productivity.
  `,
  keywords: "productivity,tasks,collaboration,management,efficiency",
  promotional_text: "Limited time: Premium features free for 30 days!",
  support_url: "https://yourapp.com/support",
  marketing_url: "https://yourapp.com",
  privacy_policy_url: "https://yourapp.com/privacy",
  
  // Screenshots and media
  screenshots: {
    ios_6_5: ["screenshot1.png", "screenshot2.png", "screenshot3.png"],
    ios_5_5: ["screenshot1_small.png", "screenshot2_small.png"],
    ios_ipad: ["ipad_screenshot1.png", "ipad_screenshot2.png"]
  },
  
  // Categories
  primary_category: "PRODUCTIVITY",
  secondary_category: "BUSINESS",
  
  // Age rating
  age_rating: "4+",
  
  // Pricing
  price_tier: "Free",
  
  // Release notes
  whats_new: `
    Version 2.1.0
    
    • New dark mode theme
    • Improved performance
    • Bug fixes and stability improvements
    • Enhanced user interface
  `
};
```

### Google Play Store Configuration
```json
{
  "app_name": "Your Awesome App",
  "short_description": "Boost your productivity with our innovative task management app",
  "full_description": "Transform your daily workflow with our comprehensive productivity suite featuring task management, team collaboration, and advanced analytics. Perfect for individuals and teams seeking efficiency.",
  "category": "PRODUCTIVITY",
  "content_rating": "Everyone",
  "price": "Free",
  "in_app_products": true,
  "ads": false,
  "target_sdk_version": 33,
  "privacy_policy": "https://yourapp.com/privacy",
  "website": "https://yourapp.com",
  "email": "support@yourapp.com",
  "phone": "+1-555-123-4567",
  "release_notes": {
    "en-US": "Bug fixes and performance improvements",
    "es-ES": "Correcciones de errores y mejoras de rendimiento"
  },
  "screenshots": {
    "phone": ["phone1.png", "phone2.png", "phone3.png"],
    "tablet": ["tablet1.png", "tablet2.png"],
    "tv": ["tv1.png", "tv2.png"]
  },
  "feature_graphic": "feature_graphic.png",
  "icon": "icon.png"
}
```

## Release Management

### Version Management Script
```javascript
// scripts/version-manager.js
const fs = require('fs');
const { execSync } = require('child_process');

class VersionManager {
  static getCurrentVersion() {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    return packageJson.version;
  }

  static bumpVersion(type = 'patch') {
    const validTypes = ['major', 'minor', 'patch'];
    if (!validTypes.includes(type)) {
      throw new Error(`Invalid bump type: ${type}`);
    }

    // Bump npm version
    execSync(`npm version ${type} --no-git-tag-version`, { stdio: 'inherit' });

    const newVersion = this.getCurrentVersion();
    
    // Update iOS version
    this.updateIOSVersion(newVersion);
    
    // Update Android version
    this.updateAndroidVersion(newVersion);
    
    // Create git tag
    execSync(`git add .`);
    execSync(`git commit -m "Bump version to ${newVersion}"`);
    execSync(`git tag v${newVersion}`);
    
    console.log(`✅ Version bumped to ${newVersion}`);
    return newVersion;
  }

  static updateIOSVersion(version) {
    const infoPlistPath = 'ios/YourApp/Info.plist';
    let infoPlist = fs.readFileSync(infoPlistPath, 'utf8');
    
    // Update CFBundleShortVersionString
    infoPlist = infoPlist.replace(
      /<key>CFBundleShortVersionString<\/key>\s*<string>[^<]*<\/string>/,
      `<key>CFBundleShortVersionString</key>\n\t<string>${version}</string>`
    );
    
    // Update CFBundleVersion (build number)
    const buildNumber = Date.now().toString();
    infoPlist = infoPlist.replace(
      /<key>CFBundleVersion<\/key>\s*<string>[^<]*<\/string>/,
      `<key>CFBundleVersion</key>\n\t<string>${buildNumber}</string>`
    );
    
    fs.writeFileSync(infoPlistPath, infoPlist);
    console.log(`✅ Updated iOS version to ${version} (${buildNumber})`);
  }

  static updateAndroidVersion(version) {
    const buildGradlePath = 'android/app/build.gradle';
    let buildGradle = fs.readFileSync(buildGradlePath, 'utf8');
    
    // Update versionName
    buildGradle = buildGradle.replace(
      /versionName\s+"[^"]*"/,
      `versionName "${version}"`
    );
    
    // Update versionCode
    const versionCode = Math.floor(Date.now() / 1000);
    buildGradle = buildGradle.replace(
      /versionCode\s+\d+/,
      `versionCode ${versionCode}`
    );
    
    fs.writeFileSync(buildGradlePath, buildGradle);
    console.log(`✅ Updated Android version to ${version} (${versionCode})`);
  }

  static createReleaseNotes(version) {
    const releaseNotesPath = `release-notes/v${version}.md`;
    const template = `# Release Notes - v${version}

## 🚀 New Features
- Feature 1
- Feature 2

## 🐛 Bug Fixes
- Fix 1
- Fix 2

## 🔧 Improvements
- Improvement 1
- Improvement 2

## 📱 Platform Specific
### iOS
- iOS specific changes

### Android
- Android specific changes

## 🚨 Breaking Changes
- None

## 📋 Checklist
- [ ] Tested on iOS
- [ ] Tested on Android
- [ ] Updated documentation
- [ ] Updated screenshots
- [ ] Prepared release notes
`;

    fs.writeFileSync(releaseNotesPath, template);
    console.log(`✅ Created release notes template: ${releaseNotesPath}`);
  }
}

// CLI usage
if (require.main === module) {
  const [,, command, type] = process.argv;
  
  switch (command) {
    case 'bump':
      VersionManager.bumpVersion(type || 'patch');
      break;
    case 'notes':
      const version = VersionManager.getCurrentVersion();
      VersionManager.createReleaseNotes(version);
      break;
    default:
      console.log('Usage: node version-manager.js <bump|notes> [major|minor|patch]');
  }
}

module.exports = VersionManager;
```

### Release Automation Script
```bash
#!/bin/bash
# scripts/release.sh

set -e

VERSION_TYPE=${1:-patch}
PLATFORM=${2:-all}

echo "🚀 Starting release process..."
echo "Version type: $VERSION_TYPE"
echo "Platform: $PLATFORM"

# Ensure we're on main branch
git checkout main
git pull origin main

# Run tests
echo "🧪 Running tests..."
npm test

# Bump version
echo "⬆️ Bumping version..."
node scripts/version-manager.js bump $VERSION_TYPE

NEW_VERSION=$(node -p "require('./package.json').version")
echo "New version: $NEW_VERSION"

# Build and deploy based on platform
case $PLATFORM in
  "ios")
    echo "📱 Building iOS..."
    eas build --platform ios --non-interactive
    ;;
  "android")
    echo "🤖 Building Android..."
    eas build --platform android --non-interactive
    ;;
  "all")
    echo "📱🤖 Building for all platforms..."
    eas build --platform all --non-interactive
    ;;
esac

# Push tags
echo "🏷️ Pushing tags..."
git push origin main --tags

# Create release notes
echo "📝 Creating release notes..."
node scripts/version-manager.js notes

echo "✅ Release $NEW_VERSION completed!"
echo "Next steps:"
echo "1. Update release notes in release-notes/v$NEW_VERSION.md"
echo "2. Submit builds to app stores"
echo "3. Publish OTA update: eas update --branch production"
```

---

*Continue to: [17-react-native-advanced.md](./17-react-native-advanced.md)*

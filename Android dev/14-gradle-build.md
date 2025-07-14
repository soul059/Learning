# Gradle Build System

## Table of Contents
- [Gradle Overview](#gradle-overview)
- [Build Configuration](#build-configuration)
- [Dependencies Management](#dependencies-management)
- [Build Types and Flavors](#build-types-and-flavors)
- [Build Optimization](#build-optimization)
- [Custom Tasks](#custom-tasks)
- [Multi-Module Projects](#multi-module-projects)
- [Advanced Configuration](#advanced-configuration)

## Gradle Overview

### What is Gradle?
Gradle is the official build system for Android that automates and manages the build process. It uses Groovy/Kotlin DSL for configuration files.

### Key Concepts
- **Project**: A collection of tasks and configurations
- **Tasks**: Units of work that can be executed
- **Plugins**: Extensions that add functionality
- **Dependencies**: External libraries and modules

### Gradle Files Structure
```
MyProject/
├── build.gradle (Project-level)
├── settings.gradle
├── gradle.properties
├── app/
│   └── build.gradle (Module-level)
└── gradle/
    └── wrapper/
        ├── gradle-wrapper.jar
        └── gradle-wrapper.properties
```

## Build Configuration

### Project-level build.gradle
```gradle
// Top-level build file where you can add configuration options common to all sub-projects/modules

buildscript {
    ext {
        // Define versions
        kotlin_version = '1.9.10'
        gradle_version = '8.1.2'
        compileSdk = 34
        targetSdk = 34
        minSdk = 21
        
        // Library versions
        appcompat_version = '1.6.1'
        material_version = '1.10.0'
        constraint_layout_version = '2.1.4'
        lifecycle_version = '2.7.0'
        room_version = '2.5.0'
        retrofit_version = '2.9.0'
        glide_version = '4.15.1'
    }
    
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    
    dependencies {
        classpath "com.android.tools.build:gradle:$gradle_version"
        classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"
        
        // Additional plugins
        classpath 'com.google.gms:google-services:4.4.0'
        classpath 'com.google.firebase:firebase-crashlytics-gradle:2.9.9'
        classpath 'androidx.navigation:navigation-safe-args-gradle-plugin:2.7.4'
    }
}

plugins {
    id 'com.android.application' version '8.1.2' apply false
    id 'com.android.library' version '8.1.2' apply false
    id 'org.jetbrains.kotlin.android' version '1.9.10' apply false
}

allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}

task clean(type: Delete) {
    delete rootProject.buildDir
}

// Configuration for all Android modules
subprojects {
    afterEvaluate { project ->
        if (project.hasProperty('android')) {
            android {
                compileSdk rootProject.ext.compileSdk
                
                defaultConfig {
                    minSdk rootProject.ext.minSdk
                    targetSdk rootProject.ext.targetSdk
                    
                    testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
                }
                
                compileOptions {
                    sourceCompatibility JavaVersion.VERSION_1_8
                    targetCompatibility JavaVersion.VERSION_1_8
                }
                
                if (project.hasProperty('kotlinOptions')) {
                    kotlinOptions {
                        jvmTarget = '1.8'
                    }
                }
            }
        }
    }
}
```

### Module-level build.gradle
```gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
    id 'kotlin-kapt'
    id 'kotlin-parcelize'
    id 'androidx.navigation.safeargs.kotlin'
    id 'com.google.gms.google-services'
    id 'com.google.firebase.crashlytics'
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
        
        // Build configuration fields
        buildConfigField "String", "API_BASE_URL", '"https://api.example.com/"'
        buildConfigField "boolean", "DEBUG_MODE", "true"
        
        // Resource values
        resValue "string", "app_name", "My App"
        
        // ProGuard files
        proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        
        // Vector drawable support
        vectorDrawables.useSupportLibrary = true
        
        // MultiDex support
        multiDexEnabled true
        
        // NDK configuration
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a', 'x86', 'x86_64'
        }
    }

    signingConfigs {
        debug {
            storeFile file('debug.keystore')
            storePassword 'android'
            keyAlias 'androiddebugkey'
            keyPassword 'android'
        }
        
        release {
            storeFile file('release.keystore')
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias System.getenv("KEY_ALIAS")
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }

    buildTypes {
        debug {
            debuggable true
            minifyEnabled false
            shrinkResources false
            signingConfig signingConfigs.debug
            applicationIdSuffix ".debug"
            versionNameSuffix "-DEBUG"
            
            buildConfigField "String", "API_BASE_URL", '"https://api-dev.example.com/"'
            resValue "string", "app_name", "My App Debug"
        }
        
        release {
            debuggable false
            minifyEnabled true
            shrinkResources true
            signingConfig signingConfigs.release
            
            buildConfigField "String", "API_BASE_URL", '"https://api.example.com/"'
            resValue "string", "app_name", "My App"
        }
        
        staging {
            initWith debug
            debuggable false
            applicationIdSuffix ".staging"
            versionNameSuffix "-STAGING"
            
            buildConfigField "String", "API_BASE_URL", '"https://api-staging.example.com/"'
            resValue "string", "app_name", "My App Staging"
        }
    }
    
    flavorDimensions "version", "environment"
    
    productFlavors {
        free {
            dimension "version"
            applicationIdSuffix ".free"
            versionNameSuffix "-free"
            
            buildConfigField "boolean", "IS_PREMIUM", "false"
            resValue "string", "app_flavor", "Free"
        }
        
        premium {
            dimension "version"
            applicationIdSuffix ".premium"
            versionNameSuffix "-premium"
            
            buildConfigField "boolean", "IS_PREMIUM", "true"
            resValue "string", "app_flavor", "Premium"
        }
        
        development {
            dimension "environment"
            buildConfigField "String", "ENVIRONMENT", '"development"'
        }
        
        production {
            dimension "environment"
            buildConfigField "String", "ENVIRONMENT", '"production"'
        }
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    
    kotlinOptions {
        jvmTarget = '1.8'
        freeCompilerArgs += [
            "-Xopt-in=kotlin.RequiresOptIn",
            "-Xopt-in=kotlinx.coroutines.ExperimentalCoroutinesApi"
        ]
    }
    
    buildFeatures {
        viewBinding true
        dataBinding true
        buildConfig true
        compose true
    }
    
    composeOptions {
        kotlinCompilerExtensionVersion '1.5.4'
    }
    
    packagingOptions {
        resources {
            excludes += [
                '/META-INF/{AL2.0,LGPL2.1}',
                '/META-INF/DEPENDENCIES',
                '/META-INF/LICENSE',
                '/META-INF/LICENSE.txt',
                '/META-INF/NOTICE',
                '/META-INF/NOTICE.txt'
            ]
        }
    }
    
    lint {
        checkReleaseBuilds false
        abortOnError false
        disable 'InvalidPackage'
        warning 'ImpliedQuantity', 'MissingQuantity'
    }
    
    testOptions {
        unitTests {
            includeAndroidResources = true
            returnDefaultValues = true
        }
        
        animationsDisabled = true
    }
}

dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    
    // Kotlin
    implementation "org.jetbrains.kotlin:kotlin-stdlib:$kotlin_version"
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // AndroidX Core
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'androidx.activity:activity-ktx:1.8.0'
    implementation 'androidx.fragment:fragment-ktx:1.6.2'
    implementation 'androidx.multidex:multidex:2.0.1'
    
    // UI Components
    implementation 'com.google.android.material:material:1.10.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    implementation 'androidx.recyclerview:recyclerview:1.3.2'
    implementation 'androidx.cardview:cardview:1.0.0'
    implementation 'androidx.swiperefreshlayout:swiperefreshlayout:1.1.0'
    
    // ViewModel and LiveData
    implementation 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0'
    implementation 'androidx.lifecycle:lifecycle-livedata-ktx:2.7.0'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.7.0'
    
    // Navigation
    implementation 'androidx.navigation:navigation-fragment-ktx:2.7.4'
    implementation 'androidx.navigation:navigation-ui-ktx:2.7.4'
    
    // Room Database
    implementation 'androidx.room:room-runtime:2.5.0'
    implementation 'androidx.room:room-ktx:2.5.0'
    kapt 'androidx.room:room-compiler:2.5.0'
    
    // Networking
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.11.0'
    
    // Image Loading
    implementation 'com.github.bumptech.glide:glide:4.15.1'
    kapt 'com.github.bumptech.glide:compiler:4.15.1'
    
    // Dependency Injection
    implementation 'com.google.dagger:dagger:2.48'
    implementation 'com.google.dagger:dagger-android:2.48'
    implementation 'com.google.dagger:dagger-android-support:2.48'
    kapt 'com.google.dagger:dagger-compiler:2.48'
    kapt 'com.google.dagger:dagger-android-processor:2.48'
    
    // Firebase
    implementation platform('com.google.firebase:firebase-bom:32.3.1')
    implementation 'com.google.firebase:firebase-analytics-ktx'
    implementation 'com.google.firebase:firebase-crashlytics-ktx'
    implementation 'com.google.firebase:firebase-messaging-ktx'
    
    // Compose (if using)
    implementation platform('androidx.compose:compose-bom:2023.10.01')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.ui:ui-tooling-preview'
    implementation 'androidx.compose.material3:material3'
    implementation 'androidx.activity:activity-compose:1.8.0'
    debugImplementation 'androidx.compose.ui:ui-tooling'
    
    // Testing
    testImplementation 'junit:junit:4.13.2'
    testImplementation 'org.mockito:mockito-core:5.1.1'
    testImplementation 'androidx.arch.core:core-testing:2.2.0'
    testImplementation 'org.robolectric:robolectric:4.10.3'
    
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
    androidTestImplementation 'androidx.test.espresso:espresso-contrib:3.5.1'
    androidTestImplementation 'androidx.test:runner:1.5.2'
    androidTestImplementation 'androidx.test:rules:1.5.0'
    
    // Debug tools
    debugImplementation 'com.squareup.leakcanary:leakcanary-android:2.12'
    debugImplementation 'com.facebook.flipper:flipper:0.230.0'
    debugImplementation 'com.facebook.soloader:soloader:0.10.5'
    debugImplementation 'com.facebook.flipper:flipper-network-plugin:0.230.0'
}

// Apply plugins at the end
apply plugin: 'com.google.gms.google-services'
apply plugin: 'com.google.firebase.crashlytics'
```

### settings.gradle
```gradle
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
        maven { url 'https://maven.google.com' }
    }
}

rootProject.name = "My Application"
include ':app'
include ':core'
include ':data'
include ':domain'
include ':feature:auth'
include ':feature:profile'
```

### gradle.properties
```properties
# Project-wide Gradle settings

# IDE (e.g. Android Studio) users:
# Gradle settings configured through the IDE *will override*
# any settings specified in this file.

# For more details on how to configure your build environment visit
# http://www.gradle.org/docs/current/userguide/build_environment.html

# Specifies the JVM arguments used for the daemon process.
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8

# When configured, Gradle will run in incubating parallel mode.
org.gradle.parallel=true

# Enable Gradle Daemon
org.gradle.daemon=true

# Enable configuration cache
org.gradle.configuration-cache=true

# Enable build cache
org.gradle.caching=true

# AndroidX package structure
android.useAndroidX=true

# Kotlin code style
kotlin.code.style=official

# Enables namespacing of each library's R class
android.nonTransitiveRClass=true

# Enable incremental compilation
kotlin.incremental=true

# Enable experimental features
android.experimental.enableArtProfiles=true

# R8 full mode
android.enableR8.fullMode=true

# Disable BuildConfig generation for libraries
android.defaults.buildfeatures.buildconfig=false

# Disable AAPT2 PNG crunching
android.enablePngCrunchInReleaseBuilds=false

# Memory optimization
org.gradle.workers.max=4

# API keys (use environment variables or local.properties for sensitive data)
API_KEY=your_api_key_here
DATABASE_URL=your_database_url_here
```

## Dependencies Management

### Version Catalogs (Gradle 7.0+)
```toml
# gradle/libs.versions.toml
[versions]
agp = "8.1.2"
kotlin = "1.9.10"
coreKtx = "1.12.0"
appcompat = "1.6.1"
material = "1.10.0"
constraintlayout = "2.1.4"
lifecycle = "2.7.0"
navigation = "2.7.4"
room = "2.5.0"
retrofit = "2.9.0"
glide = "4.15.1"
junit = "4.13.2"
androidxJunit = "1.1.5"
espresso = "3.5.1"

[libraries]
android-gradlePlugin = { group = "com.android.tools.build", name = "gradle", version.ref = "agp" }
kotlin-gradlePlugin = { group = "org.jetbrains.kotlin", name = "kotlin-gradle-plugin", version.ref = "kotlin" }

androidx-core-ktx = { group = "androidx.core", name = "core-ktx", version.ref = "coreKtx" }
androidx-appcompat = { group = "androidx.appcompat", name = "appcompat", version.ref = "appcompat" }
material = { group = "com.google.android.material", name = "material", version.ref = "material" }
androidx-constraintlayout = { group = "androidx.constraintlayout", name = "constraintlayout", version.ref = "constraintlayout" }

androidx-lifecycle-viewmodel-ktx = { group = "androidx.lifecycle", name = "lifecycle-viewmodel-ktx", version.ref = "lifecycle" }
androidx-lifecycle-livedata-ktx = { group = "androidx.lifecycle", name = "lifecycle-livedata-ktx", version.ref = "lifecycle" }

androidx-navigation-fragment-ktx = { group = "androidx.navigation", name = "navigation-fragment-ktx", version.ref = "navigation" }
androidx-navigation-ui-ktx = { group = "androidx.navigation", name = "navigation-ui-ktx", version.ref = "navigation" }

androidx-room-runtime = { group = "androidx.room", name = "room-runtime", version.ref = "room" }
androidx-room-ktx = { group = "androidx.room", name = "room-ktx", version.ref = "room" }
androidx-room-compiler = { group = "androidx.room", name = "room-compiler", version.ref = "room" }

retrofit = { group = "com.squareup.retrofit2", name = "retrofit", version.ref = "retrofit" }
retrofit-converter-gson = { group = "com.squareup.retrofit2", name = "converter-gson", version.ref = "retrofit" }

glide = { group = "com.github.bumptech.glide", name = "glide", version.ref = "glide" }
glide-compiler = { group = "com.github.bumptech.glide", name = "compiler", version.ref = "glide" }

junit = { group = "junit", name = "junit", version.ref = "junit" }
androidx-junit = { group = "androidx.test.ext", name = "junit", version.ref = "androidxJunit" }
androidx-espresso-core = { group = "androidx.test.espresso", name = "espresso-core", version.ref = "espresso" }

[bundles]
lifecycle = ["androidx-lifecycle-viewmodel-ktx", "androidx-lifecycle-livedata-ktx"]
navigation = ["androidx-navigation-fragment-ktx", "androidx-navigation-ui-ktx"]
room = ["androidx-room-runtime", "androidx-room-ktx"]
retrofit = ["retrofit", "retrofit-converter-gson"]
testing = ["junit", "androidx-junit", "androidx-espresso-core"]

[plugins]
android-application = { id = "com.android.application", version.ref = "agp" }
android-library = { id = "com.android.library", version.ref = "agp" }
kotlin-android = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }
```

### Using Version Catalogs
```gradle
// app/build.gradle
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

dependencies {
    implementation libs.androidx.core.ktx
    implementation libs.androidx.appcompat
    implementation libs.material
    implementation libs.androidx.constraintlayout
    
    implementation libs.bundles.lifecycle
    implementation libs.bundles.navigation
    implementation libs.bundles.room
    kapt libs.androidx.room.compiler
    
    implementation libs.bundles.retrofit
    
    implementation libs.glide
    kapt libs.glide.compiler
    
    testImplementation libs.bundles.testing
}
```

### Dependency Resolution Strategy
```gradle
// Project-level build.gradle
allprojects {
    configurations.all {
        resolutionStrategy {
            // Force specific versions
            force 'com.squareup.okhttp3:okhttp:4.11.0'
            
            // Fail on version conflict
            failOnVersionConflict()
            
            // Cache changing modules for 10 minutes
            cacheDynamicVersionsFor 10, 'minutes'
            
            // Don't cache changing modules
            cacheChangingModulesFor 0, 'seconds'
            
            // Substitute modules
            substitute module('org.apache.commons:commons-lang3:3.0') with module('org.apache.commons:commons-lang3:3.12.0')
        }
    }
}
```

## Build Types and Flavors

### Advanced Build Configuration
```gradle
android {
    buildTypes {
        debug {
            debuggable true
            minifyEnabled false
            shrinkResources false
            testCoverageEnabled true
            
            // Debug-specific ProGuard rules
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules-debug.pro'
            
            // Custom build config fields
            buildConfigField "String", "LOG_LEVEL", '"DEBUG"'
            buildConfigField "boolean", "ENABLE_LOGGING", "true"
            buildConfigField "int", "NETWORK_TIMEOUT", "30"
            
            // Custom resource values
            resValue "string", "app_name", "MyApp Debug"
            resValue "string", "content_authority", "com.example.myapp.debug.provider"
            
            // Manifest placeholders
            manifestPlaceholders = [
                appIcon: "@mipmap/ic_launcher_debug",
                appRoundIcon: "@mipmap/ic_launcher_debug_round"
            ]
        }
        
        release {
            debuggable false
            minifyEnabled true
            shrinkResources true
            zipAlignEnabled true
            
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            
            buildConfigField "String", "LOG_LEVEL", '"ERROR"'
            buildConfigField "boolean", "ENABLE_LOGGING", "false"
            buildConfigField "int", "NETWORK_TIMEOUT", "10"
            
            resValue "string", "app_name", "MyApp"
            resValue "string", "content_authority", "com.example.myapp.provider"
            
            manifestPlaceholders = [
                appIcon: "@mipmap/ic_launcher",
                appRoundIcon: "@mipmap/ic_launcher_round"
            ]
        }
        
        benchmark {
            initWith release
            debuggable false
            signingConfig signingConfigs.debug
            matchingFallbacks = ['release']
            
            buildConfigField "boolean", "ENABLE_BENCHMARKING", "true"
        }
    }
    
    productFlavors {
        free {
            dimension "tier"
            applicationIdSuffix ".free"
            versionNameSuffix "-free"
            
            buildConfigField "boolean", "IS_PREMIUM", "false"
            buildConfigField "int", "MAX_USERS", "5"
            
            resValue "string", "tier_name", "Free"
            resValue "color", "brand_color", "#FF9800"
        }
        
        premium {
            dimension "tier"
            applicationIdSuffix ".premium"
            versionNameSuffix "-premium"
            
            buildConfigField "boolean", "IS_PREMIUM", "true"
            buildConfigField "int", "MAX_USERS", "100"
            
            resValue "string", "tier_name", "Premium"
            resValue "color", "brand_color", "#4CAF50"
        }
        
        demo {
            dimension "tier"
            applicationIdSuffix ".demo"
            versionNameSuffix "-demo"
            
            buildConfigField "boolean", "IS_PREMIUM", "false"
            buildConfigField "int", "MAX_USERS", "1"
            buildConfigField "boolean", "DEMO_MODE", "true"
            
            resValue "string", "tier_name", "Demo"
        }
        
        dev {
            dimension "environment"
            buildConfigField "String", "SERVER_URL", '"https://dev.api.example.com"'
            buildConfigField "String", "ANALYTICS_KEY", '"dev_analytics_key"'
        }
        
        staging {
            dimension "environment"
            buildConfigField "String", "SERVER_URL", '"https://staging.api.example.com"'
            buildConfigField "String", "ANALYTICS_KEY", '"staging_analytics_key"'
        }
        
        prod {
            dimension "environment"
            buildConfigField "String", "SERVER_URL", '"https://api.example.com"'
            buildConfigField "String", "ANALYTICS_KEY", '"prod_analytics_key"'
        }
    }
    
    // Variant filtering
    variantFilter { variant ->
        def names = variant.flavors*.name
        
        // Exclude certain combinations
        if (names.contains("demo") && names.contains("prod")) {
            variant.ignore = true
        }
        
        if (names.contains("free") && names.contains("staging")) {
            variant.ignore = true
        }
    }
}
```

## Build Optimization

### Performance Optimization
```gradle
// gradle.properties
org.gradle.jvmargs=-Xmx4g -XX:MaxMetaspaceSize=512m -XX:+HeapDumpOnOutOfMemoryError -Dfile.encoding=UTF-8
org.gradle.parallel=true
org.gradle.caching=true
org.gradle.daemon=true
org.gradle.configureondemand=true
kotlin.incremental=true
kotlin.incremental.usePreciseJavaTracking=true
kotlin.incremental.js=true
kotlin.incremental.js.ir=true

// Enable build features only when needed
android {
    buildFeatures {
        viewBinding = true
        dataBinding = false
        compose = false
        buildConfig = true
        aidl = false
        renderScript = false
        resValues = true
        shaders = false
    }
}

// Optimize APK size
android {
    packagingOptions {
        resources {
            excludes += [
                '/META-INF/{AL2.0,LGPL2.1}',
                '/META-INF/DEPENDENCIES',
                '/META-INF/LICENSE',
                '/META-INF/LICENSE.txt',
                '/META-INF/NOTICE',
                '/META-INF/NOTICE.txt',
                '/META-INF/*.version',
                '/META-INF/*.kotlin_module',
                'DebugProbesKt.bin'
            ]
        }
    }
    
    bundle {
        language {
            enableSplit = true
        }
        density {
            enableSplit = true
        }
        abi {
            enableSplit = true
        }
    }
}
```

### ProGuard Configuration
```pro
# proguard-rules.pro

# Keep line numbers for debugging stack traces
-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name
-renamesourcefileattribute SourceFile

# Keep all classes in main package
-keep class com.example.myapp.** { *; }

# Keep data classes
-keep @kotlinx.android.parcel.Parcelize class * { *; }
-keep class * implements android.os.Parcelable { *; }

# Keep model classes for JSON serialization
-keep class com.example.myapp.data.model.** { *; }

# Retrofit
-keepattributes Signature, InnerClasses, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations
-keepclassmembers,allowshrinking,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}
-dontwarn org.codehaus.mojo.animal_sniffer.IgnoreJRERequirement
-dontwarn javax.annotation.**
-dontwarn kotlin.Unit
-dontwarn retrofit2.KotlinExtensions
-dontwarn retrofit2.KotlinExtensions$*

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-dontwarn sun.misc.**
-keep class com.google.gson.** { *; }
-keep class * implements com.google.gson.TypeAdapterFactory
-keep class * implements com.google.gson.JsonSerializer
-keep class * implements com.google.gson.JsonDeserializer

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**
-keepnames class okhttp3.internal.publicsuffix.PublicSuffixDatabase

# Room
-keep class * extends androidx.room.RoomDatabase
-dontwarn androidx.room.paging.**

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Remove logging
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int i(...);
    public static int w(...);
    public static int d(...);
    public static int e(...);
}

# Crashlytics
-keepattributes SourceFile,LineNumberTable
-keep public class * extends java.lang.Exception
```

## Custom Tasks

### Creating Custom Gradle Tasks
```gradle
// Custom task to generate version info
task generateVersionInfo {
    doLast {
        def versionFile = file("$buildDir/generated/version.txt")
        versionFile.parentFile.mkdirs()
        versionFile.text = """
            Version Code: ${android.defaultConfig.versionCode}
            Version Name: ${android.defaultConfig.versionName}
            Build Time: ${new Date()}
            Git Commit: ${getGitCommitHash()}
        """.stripIndent()
    }
}

def getGitCommitHash() {
    try {
        def stdout = new ByteArrayOutputStream()
        exec {
            commandLine 'git', 'rev-parse', '--short', 'HEAD'
            standardOutput = stdout
        }
        return stdout.toString().trim()
    } catch (Exception e) {
        return "unknown"
    }
}

// Task to copy APK to specific directory
task copyApkToDeployment(type: Copy) {
    dependsOn assembleRelease
    
    from "$buildDir/outputs/apk/release/"
    into "$rootDir/deployment/"
    include "**/*.apk"
    
    rename { fileName ->
        "MyApp-${android.defaultConfig.versionName}-${getGitCommitHash()}.apk"
    }
}

// Task to run all checks
task runAllChecks {
    dependsOn 'check', 'lint', 'testDebugUnitTest', 'connectedDebugAndroidTest'
    
    doLast {
        println "All checks completed successfully!"
    }
}

// Task to clean and build
task cleanBuild {
    dependsOn clean, assembleDebug
    
    doFirst {
        println "Starting clean build..."
    }
    
    doLast {
        println "Clean build completed!"
    }
}

// Make assembleDebug depend on clean
assembleDebug.mustRunAfter clean

// Task to generate build report
task generateBuildReport {
    doLast {
        def reportFile = file("$buildDir/reports/build-report.txt")
        reportFile.parentFile.mkdirs()
        
        def report = """
            Build Report
            ============
            Project: ${project.name}
            Version: ${android.defaultConfig.versionName} (${android.defaultConfig.versionCode})
            Build Date: ${new Date()}
            Gradle Version: ${gradle.gradleVersion}
            Build Tools: ${android.buildToolsVersion}
            Min SDK: ${android.defaultConfig.minSdk}
            Target SDK: ${android.defaultConfig.targetSdk}
            Compile SDK: ${android.compileSdkVersion}
            
            Dependencies:
            ${configurations.implementation.allDependencies.collect { "${it.group}:${it.name}:${it.version}" }.join('\n            ')}
        """.stripIndent()
        
        reportFile.text = report
        println "Build report generated: ${reportFile.absolutePath}"
    }
}

// Automatically run certain tasks
gradle.projectsEvaluated {
    tasks.withType(JavaCompile) {
        dependsOn generateVersionInfo
    }
}

// Task hooks
tasks.whenTaskAdded { task ->
    if (task.name == 'assembleRelease') {
        task.finalizedBy generateBuildReport
    }
}

// Conditional tasks
if (project.hasProperty('runTests')) {
    task conditionalTest {
        dependsOn test
        doLast {
            println "Tests executed because 'runTests' property was set"
        }
    }
}
```

## Multi-Module Projects

### Module Structure
```gradle
// settings.gradle
include ':app'
include ':core:common'
include ':core:network'
include ':core:database'
include ':feature:auth'
include ':feature:profile'
include ':feature:dashboard'
```

### Core Module build.gradle
```gradle
// core/common/build.gradle
plugins {
    id 'com.android.library'
    id 'org.jetbrains.kotlin.android'
    id 'kotlin-kapt'
}

android {
    namespace 'com.example.core.common'
    compileSdk 34

    defaultConfig {
        minSdk 21
        targetSdk 34
        
        consumerProguardFiles "consumer-rules.pro"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    buildFeatures {
        buildConfig = true
    }
}

dependencies {
    api 'androidx.core:core-ktx:1.12.0'
    api 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // Expose to all modules that depend on this
    api 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0'
    api 'androidx.lifecycle:lifecycle-livedata-ktx:2.7.0'
    
    // Keep internal to this module
    implementation 'androidx.annotation:annotation:1.7.0'
}
```

### Feature Module build.gradle
```gradle
// feature/auth/build.gradle
plugins {
    id 'com.android.library'
    id 'org.jetbrains.kotlin.android'
    id 'kotlin-kapt'
    id 'androidx.navigation.safeargs.kotlin'
}

android {
    namespace 'com.example.feature.auth'
    compileSdk 34

    defaultConfig {
        minSdk 21
        targetSdk 34
    }
    
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation project(':core:common')
    implementation project(':core:network')
    
    implementation 'androidx.fragment:fragment-ktx:1.6.2'
    implementation 'androidx.navigation:navigation-fragment-ktx:2.7.4'
    implementation 'com.google.android.material:material:1.10.0'
}
```

### Dependency Management Across Modules
```gradle
// Project-level build.gradle
subprojects {
    afterEvaluate { project ->
        if (project.plugins.hasPlugin('com.android.library') || 
            project.plugins.hasPlugin('com.android.application')) {
            
            project.android {
                compileSdk rootProject.ext.compileSdk
                
                defaultConfig {
                    minSdk rootProject.ext.minSdk
                    targetSdk rootProject.ext.targetSdk
                }
                
                compileOptions {
                    sourceCompatibility JavaVersion.VERSION_1_8
                    targetCompatibility JavaVersion.VERSION_1_8
                }
                
                if (project.plugins.hasPlugin('org.jetbrains.kotlin.android')) {
                    kotlinOptions {
                        jvmTarget = '1.8'
                    }
                }
            }
        }
    }
}

// Common dependencies for all modules
configure(subprojects.findAll { it.name != 'app' }) {
    apply plugin: 'com.android.library'
    apply plugin: 'org.jetbrains.kotlin.android'
    
    dependencies {
        implementation project(':core:common')
        
        testImplementation 'junit:junit:4.13.2'
        androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    }
}
```

## Advanced Configuration

### Custom Build Logic Plugin
```gradle
// buildSrc/src/main/groovy/AppVersionPlugin.groovy
import org.gradle.api.Plugin
import org.gradle.api.Project

class AppVersionPlugin implements Plugin<Project> {
    @Override
    void apply(Project project) {
        project.extensions.create('appVersion', AppVersionExtension)
        
        project.task('printVersion') {
            doLast {
                def extension = project.extensions.appVersion
                println "App Version: ${extension.major}.${extension.minor}.${extension.patch}"
                println "Version Code: ${extension.versionCode}"
            }
        }
        
        project.afterEvaluate {
            def extension = project.extensions.appVersion
            
            if (project.android) {
                project.android.defaultConfig.versionName = 
                    "${extension.major}.${extension.minor}.${extension.patch}"
                project.android.defaultConfig.versionCode = extension.versionCode
            }
        }
    }
}

class AppVersionExtension {
    int major = 1
    int minor = 0
    int patch = 0
    
    int getVersionCode() {
        return major * 10000 + minor * 100 + patch
    }
}
```

### Using Custom Plugin
```gradle
// app/build.gradle
plugins {
    id 'AppVersionPlugin'
}

appVersion {
    major = 2
    minor = 1
    patch = 0
}
```

### Build Variants Configuration
```gradle
android {
    // Automatically generate all possible combinations
    flavorDimensions "tier", "environment", "region"
    
    productFlavors {
        // Tier dimension
        free { dimension "tier" }
        premium { dimension "tier" }
        
        // Environment dimension
        dev { dimension "environment" }
        staging { dimension "environment" }
        prod { dimension "environment" }
        
        // Region dimension
        us { dimension "region" }
        eu { dimension "region" }
        asia { dimension "region" }
    }
    
    // Filter unwanted variants
    variantFilter { variant ->
        def names = variant.flavors*.name
        
        // Only allow certain combinations
        def validCombinations = [
            ['free', 'dev', 'us'],
            ['free', 'staging', 'us'],
            ['free', 'prod', 'us'],
            ['premium', 'dev', 'us'],
            ['premium', 'staging', 'us'],
            ['premium', 'prod', 'us'],
            ['premium', 'prod', 'eu'],
            ['premium', 'prod', 'asia']
        ]
        
        if (!validCombinations.any { it.containsAll(names) }) {
            variant.ignore = true
        }
    }
}

// Configure variants
applicationVariants.all { variant ->
    variant.outputs.all { output ->
        def flavorNames = variant.flavorName
        def buildType = variant.buildType.name
        def versionName = variant.versionName
        
        outputFileName = "MyApp-${flavorNames}-${buildType}-${versionName}.apk"
    }
    
    // Add variant-specific tasks
    variant.tasks.configureEach { task ->
        if (task.name.startsWith('assemble')) {
            task.doLast {
                println "Built variant: ${variant.name}"
            }
        }
    }
}
```

Understanding Gradle build system is essential for managing complex Android projects. Use build types and flavors effectively, optimize build performance, and leverage custom tasks to automate your development workflow.

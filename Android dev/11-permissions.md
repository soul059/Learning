# Permissions and Security

## Table of Contents
- [Permission Types](#permission-types)
- [Declaring Permissions](#declaring-permissions)
- [Runtime Permissions](#runtime-permissions)
- [Permission Groups](#permission-groups)
- [Best Practices](#best-practices)
- [Security Guidelines](#security-guidelines)
- [Common Permission Examples](#common-permission-examples)

## Permission Types

### Normal Permissions
Automatically granted at install time. No user intervention required.

```xml
<!-- AndroidManifest.xml -->
<!-- Normal permissions -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
<uses-permission android:name="android.permission.VIBRATE" />
<uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />
```

### Dangerous Permissions
Require user approval at runtime (Android 6.0+).

```xml
<!-- Dangerous permissions -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_CONTACTS" />
<uses-permission android:name="android.permission.WRITE_CONTACTS" />
<uses-permission android:name="android.permission.SEND_SMS" />
<uses-permission android:name="android.permission.CALL_PHONE" />
```

### Special Permissions
Require special handling and user action.

```xml
<!-- Special permissions -->
<uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />
<uses-permission android:name="android.permission.WRITE_SETTINGS" />
<uses-permission android:name="android.permission.REQUEST_INSTALL_PACKAGES" />
```

## Declaring Permissions

### Basic Permission Declaration
```xml
<!-- AndroidManifest.xml -->
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.myapp">
    
    <!-- Declare permissions -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission android:name="android.permission.CAMERA" />
    
    <!-- Optional permissions -->
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission-sdk-23 android:name="android.permission.ACCESS_FINE_LOCATION" />
    
    <!-- Permission with max SDK version -->
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"
        android:maxSdkVersion="28" />
    
    <application>
        <!-- App components -->
    </application>
</manifest>
```

### Custom Permissions
```xml
<!-- Define custom permission -->
<permission
    android:name="com.example.myapp.permission.CUSTOM_PERMISSION"
    android:label="@string/custom_permission_label"
    android:description="@string/custom_permission_description"
    android:protectionLevel="dangerous" />

<!-- Use custom permission -->
<uses-permission android:name="com.example.myapp.permission.CUSTOM_PERMISSION" />

<!-- Protect component with custom permission -->
<activity
    android:name=".SecureActivity"
    android:permission="com.example.myapp.permission.CUSTOM_PERMISSION" />
```

## Runtime Permissions

### Permission Manager Class
```java
public class PermissionManager {
    
    private Activity activity;
    private Map<String, Integer> permissionRequestCodes;
    private Map<Integer, OnPermissionResultListener> permissionCallbacks;
    
    public interface OnPermissionResultListener {
        void onPermissionGranted();
        void onPermissionDenied();
        void onPermissionPermanentlyDenied();
    }
    
    public PermissionManager(Activity activity) {
        this.activity = activity;
        this.permissionRequestCodes = new HashMap<>();
        this.permissionCallbacks = new HashMap<>();
        initializePermissionCodes();
    }
    
    private void initializePermissionCodes() {
        permissionRequestCodes.put(Manifest.permission.CAMERA, 100);
        permissionRequestCodes.put(Manifest.permission.ACCESS_FINE_LOCATION, 101);
        permissionRequestCodes.put(Manifest.permission.ACCESS_COARSE_LOCATION, 102);
        permissionRequestCodes.put(Manifest.permission.RECORD_AUDIO, 103);
        permissionRequestCodes.put(Manifest.permission.READ_EXTERNAL_STORAGE, 104);
        permissionRequestCodes.put(Manifest.permission.WRITE_EXTERNAL_STORAGE, 105);
        permissionRequestCodes.put(Manifest.permission.READ_CONTACTS, 106);
        permissionRequestCodes.put(Manifest.permission.WRITE_CONTACTS, 107);
        permissionRequestCodes.put(Manifest.permission.CALL_PHONE, 108);
        permissionRequestCodes.put(Manifest.permission.SEND_SMS, 109);
    }
    
    public boolean isPermissionGranted(String permission) {
        return ContextCompat.checkSelfPermission(activity, permission) 
               == PackageManager.PERMISSION_GRANTED;
    }
    
    public void requestPermission(String permission, OnPermissionResultListener listener) {
        if (isPermissionGranted(permission)) {
            listener.onPermissionGranted();
            return;
        }
        
        Integer requestCode = permissionRequestCodes.get(permission);
        if (requestCode == null) {
            listener.onPermissionDenied();
            return;
        }
        
        permissionCallbacks.put(requestCode, listener);
        
        if (ActivityCompat.shouldShowRequestPermissionRationale(activity, permission)) {
            showPermissionRationale(permission, requestCode);
        } else {
            ActivityCompat.requestPermissions(activity, new String[]{permission}, requestCode);
        }
    }
    
    public void requestMultiplePermissions(String[] permissions, OnPermissionResultListener listener) {
        List<String> permissionsToRequest = new ArrayList<>();
        
        for (String permission : permissions) {
            if (!isPermissionGranted(permission)) {
                permissionsToRequest.add(permission);
            }
        }
        
        if (permissionsToRequest.isEmpty()) {
            listener.onPermissionGranted();
            return;
        }
        
        int requestCode = 999; // Special code for multiple permissions
        permissionCallbacks.put(requestCode, listener);
        
        ActivityCompat.requestPermissions(activity, 
            permissionsToRequest.toArray(new String[0]), requestCode);
    }
    
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        OnPermissionResultListener listener = permissionCallbacks.get(requestCode);
        if (listener == null) return;
        
        permissionCallbacks.remove(requestCode);
        
        if (requestCode == 999) {
            // Multiple permissions
            handleMultiplePermissionsResult(permissions, grantResults, listener);
        } else {
            // Single permission
            handleSinglePermissionResult(requestCode, permissions, grantResults, listener);
        }
    }
    
    private void handleSinglePermissionResult(int requestCode, String[] permissions, 
                                            int[] grantResults, OnPermissionResultListener listener) {
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            listener.onPermissionGranted();
        } else {
            String permission = permissions[0];
            if (ActivityCompat.shouldShowRequestPermissionRationale(activity, permission)) {
                listener.onPermissionDenied();
            } else {
                listener.onPermissionPermanentlyDenied();
            }
        }
    }
    
    private void handleMultiplePermissionsResult(String[] permissions, int[] grantResults, 
                                               OnPermissionResultListener listener) {
        boolean allGranted = true;
        boolean anyPermanentlyDenied = false;
        
        for (int i = 0; i < grantResults.length; i++) {
            if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                if (!ActivityCompat.shouldShowRequestPermissionRationale(activity, permissions[i])) {
                    anyPermanentlyDenied = true;
                }
            }
        }
        
        if (allGranted) {
            listener.onPermissionGranted();
        } else if (anyPermanentlyDenied) {
            listener.onPermissionPermanentlyDenied();
        } else {
            listener.onPermissionDenied();
        }
    }
    
    private void showPermissionRationale(String permission, int requestCode) {
        String message = getPermissionRationaleMessage(permission);
        
        new AlertDialog.Builder(activity)
            .setTitle("Permission Required")
            .setMessage(message)
            .setPositiveButton("Grant", (dialog, which) -> {
                ActivityCompat.requestPermissions(activity, new String[]{permission}, requestCode);
            })
            .setNegativeButton("Cancel", (dialog, which) -> {
                OnPermissionResultListener listener = permissionCallbacks.get(requestCode);
                if (listener != null) {
                    permissionCallbacks.remove(requestCode);
                    listener.onPermissionDenied();
                }
            })
            .show();
    }
    
    private String getPermissionRationaleMessage(String permission) {
        switch (permission) {
            case Manifest.permission.CAMERA:
                return "This app needs camera access to take photos.";
            case Manifest.permission.ACCESS_FINE_LOCATION:
                return "This app needs location access to show your current position.";
            case Manifest.permission.RECORD_AUDIO:
                return "This app needs microphone access to record audio.";
            case Manifest.permission.READ_EXTERNAL_STORAGE:
                return "This app needs storage access to read files.";
            case Manifest.permission.WRITE_EXTERNAL_STORAGE:
                return "This app needs storage access to save files.";
            case Manifest.permission.READ_CONTACTS:
                return "This app needs contacts access to import your contacts.";
            case Manifest.permission.CALL_PHONE:
                return "This app needs phone access to make calls.";
            default:
                return "This app needs this permission to function properly.";
        }
    }
    
    public void openAppSettings() {
        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        Uri uri = Uri.fromParts("package", activity.getPackageName(), null);
        intent.setData(uri);
        activity.startActivity(intent);
    }
}
```

### Usage in Activity
```java
public class MainActivity extends AppCompatActivity {
    
    private PermissionManager permissionManager;
    private static final String[] REQUIRED_PERMISSIONS = {
        Manifest.permission.CAMERA,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.RECORD_AUDIO
    };
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        permissionManager = new PermissionManager(this);
        
        findViewById(R.id.btnRequestCamera).setOnClickListener(v -> requestCameraPermission());
        findViewById(R.id.btnRequestLocation).setOnClickListener(v -> requestLocationPermission());
        findViewById(R.id.btnRequestMultiple).setOnClickListener(v -> requestMultiplePermissions());
    }
    
    private void requestCameraPermission() {
        permissionManager.requestPermission(Manifest.permission.CAMERA, 
            new PermissionManager.OnPermissionResultListener() {
                @Override
                public void onPermissionGranted() {
                    Toast.makeText(MainActivity.this, "Camera permission granted", Toast.LENGTH_SHORT).show();
                    openCamera();
                }
                
                @Override
                public void onPermissionDenied() {
                    Toast.makeText(MainActivity.this, "Camera permission denied", Toast.LENGTH_SHORT).show();
                }
                
                @Override
                public void onPermissionPermanentlyDenied() {
                    showPermanentlyDeniedDialog("Camera permission is required to take photos. " +
                        "Please enable it in app settings.");
                }
            });
    }
    
    private void requestLocationPermission() {
        permissionManager.requestPermission(Manifest.permission.ACCESS_FINE_LOCATION,
            new PermissionManager.OnPermissionResultListener() {
                @Override
                public void onPermissionGranted() {
                    Toast.makeText(MainActivity.this, "Location permission granted", Toast.LENGTH_SHORT).show();
                    getCurrentLocation();
                }
                
                @Override
                public void onPermissionDenied() {
                    Toast.makeText(MainActivity.this, "Location permission denied", Toast.LENGTH_SHORT).show();
                }
                
                @Override
                public void onPermissionPermanentlyDenied() {
                    showPermanentlyDeniedDialog("Location permission is required to show your position. " +
                        "Please enable it in app settings.");
                }
            });
    }
    
    private void requestMultiplePermissions() {
        permissionManager.requestMultiplePermissions(REQUIRED_PERMISSIONS,
            new PermissionManager.OnPermissionResultListener() {
                @Override
                public void onPermissionGranted() {
                    Toast.makeText(MainActivity.this, "All permissions granted", Toast.LENGTH_SHORT).show();
                    initializeApp();
                }
                
                @Override
                public void onPermissionDenied() {
                    Toast.makeText(MainActivity.this, "Some permissions denied", Toast.LENGTH_SHORT).show();
                }
                
                @Override
                public void onPermissionPermanentlyDenied() {
                    showPermanentlyDeniedDialog("This app requires all permissions to function properly. " +
                        "Please enable them in app settings.");
                }
            });
    }
    
    private void showPermanentlyDeniedDialog(String message) {
        new AlertDialog.Builder(this)
            .setTitle("Permission Required")
            .setMessage(message)
            .setPositiveButton("Open Settings", (dialog, which) -> {
                permissionManager.openAppSettings();
            })
            .setNegativeButton("Cancel", null)
            .show();
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, 
                                         @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        permissionManager.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
    
    private void openCamera() {
        // Camera functionality
        Log.d("MainActivity", "Opening camera");
    }
    
    private void getCurrentLocation() {
        // Location functionality
        Log.d("MainActivity", "Getting current location");
    }
    
    private void initializeApp() {
        // Initialize app with all permissions
        Log.d("MainActivity", "Initializing app with all permissions");
    }
}
```

## Permission Groups

### Location Permissions
```java
public class LocationPermissionHelper {
    
    public static void requestLocationPermission(Activity activity, 
                                               PermissionManager.OnPermissionResultListener listener) {
        // First try precise location
        PermissionManager permissionManager = new PermissionManager(activity);
        
        permissionManager.requestPermission(Manifest.permission.ACCESS_FINE_LOCATION,
            new PermissionManager.OnPermissionResultListener() {
                @Override
                public void onPermissionGranted() {
                    listener.onPermissionGranted();
                }
                
                @Override
                public void onPermissionDenied() {
                    // Try approximate location as fallback
                    permissionManager.requestPermission(Manifest.permission.ACCESS_COARSE_LOCATION, listener);
                }
                
                @Override
                public void onPermissionPermanentlyDenied() {
                    listener.onPermissionPermanentlyDenied();
                }
            });
    }
    
    public static boolean hasLocationPermission(Context context) {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) 
               == PackageManager.PERMISSION_GRANTED ||
               ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) 
               == PackageManager.PERMISSION_GRANTED;
    }
    
    public static boolean hasPreciseLocationPermission(Context context) {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) 
               == PackageManager.PERMISSION_GRANTED;
    }
}
```

### Storage Permissions
```java
public class StoragePermissionHelper {
    
    public static void requestStoragePermission(Activity activity, 
                                              PermissionManager.OnPermissionResultListener listener) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Android 11+ - Scoped Storage
            if (Environment.isExternalStorageManager()) {
                listener.onPermissionGranted();
            } else {
                requestManageExternalStoragePermission(activity, listener);
            }
        } else {
            // Android 10 and below
            PermissionManager permissionManager = new PermissionManager(activity);
            String[] permissions = {
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            };
            permissionManager.requestMultiplePermissions(permissions, listener);
        }
    }
    
    @TargetApi(Build.VERSION_CODES.R)
    private static void requestManageExternalStoragePermission(Activity activity,
                                                             PermissionManager.OnPermissionResultListener listener) {
        new AlertDialog.Builder(activity)
            .setTitle("Storage Permission")
            .setMessage("This app needs access to manage external storage. You will be redirected to settings.")
            .setPositiveButton("OK", (dialog, which) -> {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                Uri uri = Uri.fromParts("package", activity.getPackageName(), null);
                intent.setData(uri);
                activity.startActivity(intent);
            })
            .setNegativeButton("Cancel", (dialog, which) -> listener.onPermissionDenied())
            .show();
    }
    
    public static boolean hasStoragePermission(Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            return Environment.isExternalStorageManager();
        } else {
            return ContextCompat.checkSelfPermission(context, Manifest.permission.READ_EXTERNAL_STORAGE) 
                   == PackageManager.PERMISSION_GRANTED;
        }
    }
}
```

### Camera and Audio Permissions
```java
public class MediaPermissionHelper {
    
    public static void requestCameraAndAudioPermissions(Activity activity,
                                                      PermissionManager.OnPermissionResultListener listener) {
        PermissionManager permissionManager = new PermissionManager(activity);
        String[] permissions = {
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        };
        permissionManager.requestMultiplePermissions(permissions, listener);
    }
    
    public static boolean hasCameraPermission(Context context) {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) 
               == PackageManager.PERMISSION_GRANTED;
    }
    
    public static boolean hasAudioPermission(Context context) {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) 
               == PackageManager.PERMISSION_GRANTED;
    }
    
    public static void checkCameraAvailability(Context context, OnCameraCheckListener listener) {
        if (!hasCameraPermission(context)) {
            listener.onPermissionRequired();
            return;
        }
        
        PackageManager pm = context.getPackageManager();
        if (pm.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            listener.onCameraAvailable();
        } else {
            listener.onCameraNotAvailable();
        }
    }
    
    public interface OnCameraCheckListener {
        void onCameraAvailable();
        void onCameraNotAvailable();
        void onPermissionRequired();
    }
}
```

## Best Practices

### 1. Permission Strategy Pattern
```java
public abstract class PermissionStrategy {
    
    protected Activity activity;
    protected PermissionManager permissionManager;
    
    public PermissionStrategy(Activity activity) {
        this.activity = activity;
        this.permissionManager = new PermissionManager(activity);
    }
    
    public abstract void requestPermissions(PermissionManager.OnPermissionResultListener listener);
    public abstract boolean hasRequiredPermissions();
    public abstract String[] getRequiredPermissions();
    
    protected void showFeatureUnavailableDialog(String message) {
        new AlertDialog.Builder(activity)
            .setTitle("Feature Unavailable")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show();
    }
}

public class CameraPermissionStrategy extends PermissionStrategy {
    
    public CameraPermissionStrategy(Activity activity) {
        super(activity);
    }
    
    @Override
    public void requestPermissions(PermissionManager.OnPermissionResultListener listener) {
        permissionManager.requestPermission(Manifest.permission.CAMERA, listener);
    }
    
    @Override
    public boolean hasRequiredPermissions() {
        return permissionManager.isPermissionGranted(Manifest.permission.CAMERA);
    }
    
    @Override
    public String[] getRequiredPermissions() {
        return new String[]{Manifest.permission.CAMERA};
    }
}

public class LocationPermissionStrategy extends PermissionStrategy {
    
    public LocationPermissionStrategy(Activity activity) {
        super(activity);
    }
    
    @Override
    public void requestPermissions(PermissionManager.OnPermissionResultListener listener) {
        String[] permissions = {
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        };
        permissionManager.requestMultiplePermissions(permissions, listener);
    }
    
    @Override
    public boolean hasRequiredPermissions() {
        return permissionManager.isPermissionGranted(Manifest.permission.ACCESS_FINE_LOCATION) ||
               permissionManager.isPermissionGranted(Manifest.permission.ACCESS_COARSE_LOCATION);
    }
    
    @Override
    public String[] getRequiredPermissions() {
        return new String[]{
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        };
    }
}
```

### 2. Permission State Manager
```java
public class PermissionStateManager {
    
    private static final String PREFS_NAME = "permission_prefs";
    private static final String KEY_PERMISSION_ASKED = "permission_asked_";
    private static final String KEY_PERMISSION_DENIED = "permission_denied_";
    
    private SharedPreferences prefs;
    
    public PermissionStateManager(Context context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }
    
    public void markPermissionAsked(String permission) {
        prefs.edit().putBoolean(KEY_PERMISSION_ASKED + permission, true).apply();
    }
    
    public boolean wasPermissionAsked(String permission) {
        return prefs.getBoolean(KEY_PERMISSION_ASKED + permission, false);
    }
    
    public void markPermissionDenied(String permission) {
        prefs.edit().putBoolean(KEY_PERMISSION_DENIED + permission, true).apply();
    }
    
    public boolean wasPermissionDenied(String permission) {
        return prefs.getBoolean(KEY_PERMISSION_DENIED + permission, false);
    }
    
    public void clearPermissionState(String permission) {
        prefs.edit()
            .remove(KEY_PERMISSION_ASKED + permission)
            .remove(KEY_PERMISSION_DENIED + permission)
            .apply();
    }
    
    public boolean shouldShowRationale(Activity activity, String permission) {
        return ActivityCompat.shouldShowRequestPermissionRationale(activity, permission) ||
               (!wasPermissionAsked(permission) && !wasPermissionDenied(permission));
    }
}
```

### 3. Graceful Degradation
```java
public class FeatureManager {
    
    private Activity activity;
    private PermissionManager permissionManager;
    
    public FeatureManager(Activity activity) {
        this.activity = activity;
        this.permissionManager = new PermissionManager(activity);
    }
    
    public void enableCameraFeature(OnFeatureEnabledListener listener) {
        if (permissionManager.isPermissionGranted(Manifest.permission.CAMERA)) {
            listener.onFeatureEnabled();
        } else {
            permissionManager.requestPermission(Manifest.permission.CAMERA,
                new PermissionManager.OnPermissionResultListener() {
                    @Override
                    public void onPermissionGranted() {
                        listener.onFeatureEnabled();
                    }
                    
                    @Override
                    public void onPermissionDenied() {
                        listener.onFeatureDisabled("Camera permission required");
                    }
                    
                    @Override
                    public void onPermissionPermanentlyDenied() {
                        listener.onFeatureDisabled("Camera permission permanently denied");
                    }
                });
        }
    }
    
    public void enableLocationFeature(OnFeatureEnabledListener listener) {
        if (LocationPermissionHelper.hasLocationPermission(activity)) {
            listener.onFeatureEnabled();
        } else {
            LocationPermissionHelper.requestLocationPermission(activity,
                new PermissionManager.OnPermissionResultListener() {
                    @Override
                    public void onPermissionGranted() {
                        listener.onFeatureEnabled();
                    }
                    
                    @Override
                    public void onPermissionDenied() {
                        // Provide alternative functionality
                        listener.onFeatureEnabledWithLimitations("Location features limited");
                    }
                    
                    @Override
                    public void onPermissionPermanentlyDenied() {
                        listener.onFeatureDisabled("Location permission required");
                    }
                });
        }
    }
    
    public interface OnFeatureEnabledListener {
        void onFeatureEnabled();
        default void onFeatureEnabledWithLimitations(String message) {}
        void onFeatureDisabled(String reason);
    }
}
```

## Security Guidelines

### 1. Secure Data Storage
```java
public class SecureStorage {
    
    private SharedPreferences securePrefs;
    private static final String PREFS_NAME = "secure_prefs";
    
    public SecureStorage(Context context) {
        // Use EncryptedSharedPreferences for sensitive data
        try {
            MasterKeys.AliasSpec spec = new MasterKeys.AliasSpec.Builder(MasterKeys.AES256_GCM_SPEC)
                .setKeyScheme(MasterKeys.KeyScheme.AES256_GCM)
                .build();
            
            String masterKeyAlias = MasterKeys.getOrCreate(spec);
            
            securePrefs = EncryptedSharedPreferences.create(
                PREFS_NAME,
                masterKeyAlias,
                context,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );
        } catch (Exception e) {
            Log.e("SecureStorage", "Error creating encrypted preferences", e);
            // Fallback to regular SharedPreferences
            securePrefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        }
    }
    
    public void storeSecureData(String key, String value) {
        securePrefs.edit().putString(key, value).apply();
    }
    
    public String getSecureData(String key, String defaultValue) {
        return securePrefs.getString(key, defaultValue);
    }
    
    public void clearSecureData() {
        securePrefs.edit().clear().apply();
    }
}
```

### 2. Network Security
```java
public class NetworkSecurityManager {
    
    public static OkHttpClient createSecureHttpClient(Context context) {
        // Certificate pinning
        CertificatePinner certificatePinner = new CertificatePinner.Builder()
            .add("api.yourapp.com", "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
            .build();
        
        // Create custom trust manager
        X509TrustManager trustManager = createTrustManager(context);
        
        // Create SSL context
        SSLContext sslContext = createSSLContext(trustManager);
        
        return new OkHttpClient.Builder()
            .certificatePinner(certificatePinner)
            .sslSocketFactory(sslContext.getSocketFactory(), trustManager)
            .addInterceptor(new AuthenticationInterceptor())
            .addInterceptor(new LoggingInterceptor())
            .build();
    }
    
    private static X509TrustManager createTrustManager(Context context) {
        try {
            TrustManagerFactory factory = TrustManagerFactory.getInstance(
                TrustManagerFactory.getDefaultAlgorithm());
            factory.init((KeyStore) null);
            
            TrustManager[] trustManagers = factory.getTrustManagers();
            return (X509TrustManager) trustManagers[0];
        } catch (Exception e) {
            throw new RuntimeException("Failed to create trust manager", e);
        }
    }
    
    private static SSLContext createSSLContext(X509TrustManager trustManager) {
        try {
            SSLContext sslContext = SSLContext.getInstance("TLS");
            sslContext.init(null, new TrustManager[]{trustManager}, null);
            return sslContext;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create SSL context", e);
        }
    }
    
    private static class AuthenticationInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request originalRequest = chain.request();
            
            // Add authentication headers
            Request authenticatedRequest = originalRequest.newBuilder()
                .addHeader("Authorization", "Bearer " + getAuthToken())
                .addHeader("X-API-Key", getApiKey())
                .build();
            
            return chain.proceed(authenticatedRequest);
        }
        
        private String getAuthToken() {
            // Get secure auth token
            return "secure_token";
        }
        
        private String getApiKey() {
            // Get API key from secure storage
            return "api_key";
        }
    }
}
```

### 3. Input Validation
```java
public class InputValidator {
    
    public static boolean isValidEmail(String email) {
        return email != null && 
               Patterns.EMAIL_ADDRESS.matcher(email).matches() &&
               email.length() <= 254; // RFC 5321 limit
    }
    
    public static boolean isValidPhoneNumber(String phone) {
        return phone != null &&
               phone.matches("\\+?[0-9\\s\\-\\(\\)]{10,15}");
    }
    
    public static boolean isValidUrl(String url) {
        return url != null &&
               Patterns.WEB_URL.matcher(url).matches();
    }
    
    public static String sanitizeInput(String input) {
        if (input == null) return "";
        
        // Remove potentially dangerous characters
        return input.replaceAll("[<>\"'&]", "")
                   .trim()
                   .substring(0, Math.min(input.length(), 1000)); // Limit length
    }
    
    public static boolean isValidPassword(String password) {
        if (password == null || password.length() < 8) {
            return false;
        }
        
        boolean hasUpper = password.matches(".*[A-Z].*");
        boolean hasLower = password.matches(".*[a-z].*");
        boolean hasDigit = password.matches(".*\\d.*");
        boolean hasSpecial = password.matches(".*[!@#$%^&*()_+\\-=\\[\\]{};':\"\\\\|,.<>\\/?].*");
        
        return hasUpper && hasLower && hasDigit && hasSpecial;
    }
}
```

## Common Permission Examples

### File Access Example
```java
public class FileAccessExample {
    
    public void readFileFromStorage(Activity activity, String filePath) {
        PermissionManager permissionManager = new PermissionManager(activity);
        
        permissionManager.requestPermission(Manifest.permission.READ_EXTERNAL_STORAGE,
            new PermissionManager.OnPermissionResultListener() {
                @Override
                public void onPermissionGranted() {
                    try {
                        File file = new File(filePath);
                        if (file.exists()) {
                            // Read file content
                            FileInputStream fis = new FileInputStream(file);
                            // Process file...
                            fis.close();
                        }
                    } catch (IOException e) {
                        Log.e("FileAccess", "Error reading file", e);
                    }
                }
                
                @Override
                public void onPermissionDenied() {
                    Toast.makeText(activity, "Storage permission required to read files", 
                        Toast.LENGTH_SHORT).show();
                }
                
                @Override
                public void onPermissionPermanentlyDenied() {
                    // Guide user to settings
                }
            });
    }
}
```

### Location Access Example
```java
public class LocationAccessExample {
    
    private FusedLocationProviderClient fusedLocationClient;
    
    public LocationAccessExample(Context context) {
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(context);
    }
    
    public void getCurrentLocation(Activity activity, OnLocationListener listener) {
        LocationPermissionHelper.requestLocationPermission(activity,
            new PermissionManager.OnPermissionResultListener() {
                @Override
                public void onPermissionGranted() {
                    getLastKnownLocation(listener);
                }
                
                @Override
                public void onPermissionDenied() {
                    listener.onLocationError("Location permission denied");
                }
                
                @Override
                public void onPermissionPermanentlyDenied() {
                    listener.onLocationError("Location permission permanently denied");
                }
            });
    }
    
    @SuppressLint("MissingPermission")
    private void getLastKnownLocation(OnLocationListener listener) {
        fusedLocationClient.getLastLocation()
            .addOnSuccessListener(location -> {
                if (location != null) {
                    listener.onLocationReceived(location.getLatitude(), location.getLongitude());
                } else {
                    listener.onLocationError("Location not available");
                }
            })
            .addOnFailureListener(e -> {
                listener.onLocationError("Failed to get location: " + e.getMessage());
            });
    }
    
    public interface OnLocationListener {
        void onLocationReceived(double latitude, double longitude);
        void onLocationError(String error);
    }
}
```

### Camera Access Example
```java
public class CameraAccessExample {
    
    public void openCamera(Activity activity, OnCameraListener listener) {
        MediaPermissionHelper.requestCameraAndAudioPermissions(activity,
            new PermissionManager.OnPermissionResultListener() {
                @Override
                public void onPermissionGranted() {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    if (cameraIntent.resolveActivity(activity.getPackageManager()) != null) {
                        listener.onCameraReady(cameraIntent);
                    } else {
                        listener.onCameraError("No camera app available");
                    }
                }
                
                @Override
                public void onPermissionDenied() {
                    listener.onCameraError("Camera permission denied");
                }
                
                @Override
                public void onPermissionPermanentlyDenied() {
                    listener.onCameraError("Camera permission permanently denied");
                }
            });
    }
    
    public interface OnCameraListener {
        void onCameraReady(Intent cameraIntent);
        void onCameraError(String error);
    }
}
```

Understanding permissions and security is crucial for creating trustworthy Android applications. Always request permissions at the right time, provide clear explanations to users, and implement proper security measures to protect user data.

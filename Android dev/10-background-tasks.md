# Background Tasks and Services

## Table of Contents
- [Overview](#overview)
- [AsyncTask (Deprecated)](#asynctask-deprecated)
- [Thread and Handler](#thread-and-handler)
- [ExecutorService](#executorservice)
- [Services](#services)
- [IntentService](#intentservice)
- [JobScheduler](#jobscheduler)
- [WorkManager](#workmanager)
- [Foreground Services](#foreground-services)
- [Best Practices](#best-practices)

## Overview

Android provides several mechanisms for performing background tasks:

1. **AsyncTask** (Deprecated in API 30) - Simple background operations
2. **Thread + Handler** - Basic threading with UI updates
3. **ExecutorService** - Thread pool management
4. **Services** - Long-running background operations
5. **IntentService** - Simple service for handling tasks sequentially
6. **JobScheduler** - System-managed background jobs
7. **WorkManager** - Modern solution for deferrable, guaranteed background work

## AsyncTask (Deprecated)

⚠️ **Note**: AsyncTask is deprecated since API level 30. Use ExecutorService or WorkManager instead.

### Basic AsyncTask Implementation
```java
public class DataDownloadTask extends AsyncTask<String, Integer, String> {
    
    private WeakReference<Context> contextRef;
    private ProgressBar progressBar;
    private OnTaskCompleteListener listener;
    
    public interface OnTaskCompleteListener {
        void onSuccess(String result);
        void onError(String error);
        void onProgress(int progress);
    }
    
    public DataDownloadTask(Context context, ProgressBar progressBar, OnTaskCompleteListener listener) {
        this.contextRef = new WeakReference<>(context);
        this.progressBar = progressBar;
        this.listener = listener;
    }
    
    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        // Run on UI thread before background work starts
        if (progressBar != null) {
            progressBar.setVisibility(View.VISIBLE);
            progressBar.setProgress(0);
        }
    }
    
    @Override
    protected String doInBackground(String... urls) {
        try {
            String url = urls[0];
            
            // Simulate download with progress
            for (int i = 0; i <= 100; i += 10) {
                Thread.sleep(100); // Simulate work
                publishProgress(i);
                
                // Check if task was cancelled
                if (isCancelled()) {
                    return null;
                }
            }
            
            // Perform actual download
            return downloadData(url);
            
        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }
    }
    
    @Override
    protected void onProgressUpdate(Integer... progress) {
        super.onProgressUpdate(progress);
        // Run on UI thread to update progress
        if (progressBar != null) {
            progressBar.setProgress(progress[0]);
        }
        
        if (listener != null) {
            listener.onProgress(progress[0]);
        }
    }
    
    @Override
    protected void onPostExecute(String result) {
        super.onPostExecute(result);
        // Run on UI thread when background work completes
        if (progressBar != null) {
            progressBar.setVisibility(View.GONE);
        }
        
        if (listener != null) {
            if (result.startsWith("Error:")) {
                listener.onError(result);
            } else {
                listener.onSuccess(result);
            }
        }
    }
    
    @Override
    protected void onCancelled() {
        super.onCancelled();
        // Called when task is cancelled
        if (progressBar != null) {
            progressBar.setVisibility(View.GONE);
        }
    }
    
    private String downloadData(String url) {
        // Implement actual download logic
        return "Downloaded data from " + url;
    }
}

// Usage
DataDownloadTask task = new DataDownloadTask(this, progressBar, 
    new DataDownloadTask.OnTaskCompleteListener() {
        @Override
        public void onSuccess(String result) {
            Toast.makeText(MainActivity.this, "Success: " + result, Toast.LENGTH_SHORT).show();
        }
        
        @Override
        public void onError(String error) {
            Toast.makeText(MainActivity.this, error, Toast.LENGTH_SHORT).show();
        }
        
        @Override
        public void onProgress(int progress) {
            Log.d("Download", "Progress: " + progress + "%");
        }
    });

task.execute("https://api.example.com/data");
```

## Thread and Handler

### Basic Thread with Handler
```java
public class ThreadHandler {
    
    private Handler mainHandler;
    
    public ThreadHandler() {
        mainHandler = new Handler(Looper.getMainLooper());
    }
    
    public void performBackgroundTask() {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                // Background work
                String result = doHeavyWork();
                
                // Update UI on main thread
                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        updateUI(result);
                    }
                });
            }
        });
        
        thread.start();
    }
    
    private String doHeavyWork() {
        // Simulate heavy work
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return "Work completed";
    }
    
    private void updateUI(String result) {
        // Update UI components
        Log.d("Thread", "UI updated with: " + result);
    }
}
```

### HandlerThread
```java
public class BackgroundProcessor {
    
    private HandlerThread handlerThread;
    private Handler backgroundHandler;
    private Handler mainHandler;
    
    public void start() {
        handlerThread = new HandlerThread("BackgroundProcessor");
        handlerThread.start();
        
        backgroundHandler = new Handler(handlerThread.getLooper());
        mainHandler = new Handler(Looper.getMainLooper());
    }
    
    public void processData(String data) {
        backgroundHandler.post(new Runnable() {
            @Override
            public void run() {
                // Process data in background
                String result = processInBackground(data);
                
                // Post result to main thread
                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        onDataProcessed(result);
                    }
                });
            }
        });
    }
    
    public void stop() {
        if (handlerThread != null) {
            handlerThread.quitSafely();
            try {
                handlerThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    
    private String processInBackground(String data) {
        // Heavy processing
        return "Processed: " + data;
    }
    
    private void onDataProcessed(String result) {
        // Update UI
        Log.d("Processor", result);
    }
}
```

## ExecutorService

### ThreadPoolExecutor Example
```java
public class TaskExecutor {
    
    private ExecutorService executorService;
    private Handler mainHandler;
    
    public TaskExecutor() {
        // Create thread pool
        executorService = Executors.newFixedThreadPool(4);
        mainHandler = new Handler(Looper.getMainLooper());
    }
    
    public <T> void executeTask(Callable<T> task, OnTaskListener<T> listener) {
        Future<T> future = executorService.submit(task);
        
        // Monitor the task in a separate thread
        executorService.execute(() -> {
            try {
                T result = future.get();
                
                // Post result to main thread
                mainHandler.post(() -> {
                    if (listener != null) {
                        listener.onSuccess(result);
                    }
                });
                
            } catch (Exception e) {
                mainHandler.post(() -> {
                    if (listener != null) {
                        listener.onError(e.getMessage());
                    }
                });
            }
        });
    }
    
    public void executeRunnable(Runnable task) {
        executorService.execute(task);
    }
    
    public void shutdown() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
    
    public interface OnTaskListener<T> {
        void onSuccess(T result);
        void onError(String error);
    }
}

// Usage
TaskExecutor taskExecutor = new TaskExecutor();

// Execute a Callable task
taskExecutor.executeTask(() -> {
    // Background work
    Thread.sleep(2000);
    return "Task completed";
}, new TaskExecutor.OnTaskListener<String>() {
    @Override
    public void onSuccess(String result) {
        Log.d("Task", "Success: " + result);
    }
    
    @Override
    public void onError(String error) {
        Log.e("Task", "Error: " + error);
    }
});

// Execute a Runnable task
taskExecutor.executeRunnable(() -> {
    // Background work without return value
    Log.d("Task", "Background work completed");
});
```

## Services

### Basic Service
```java
public class MyService extends Service {
    
    private boolean isRunning = false;
    private Thread serviceThread;
    
    @Override
    public void onCreate() {
        super.onCreate();
        Log.d("MyService", "Service created");
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d("MyService", "Service started");
        
        if (!isRunning) {
            isRunning = true;
            startBackgroundWork();
        }
        
        // Return sticky to restart service if killed
        return START_STICKY;
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        // Return null for started service
        return null;
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.d("MyService", "Service destroyed");
        isRunning = false;
        
        if (serviceThread != null) {
            serviceThread.interrupt();
        }
    }
    
    private void startBackgroundWork() {
        serviceThread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (isRunning && !Thread.currentThread().isInterrupted()) {
                    try {
                        // Perform background work
                        performWork();
                        Thread.sleep(5000); // Wait 5 seconds
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        });
        
        serviceThread.start();
    }
    
    private void performWork() {
        Log.d("MyService", "Performing background work: " + System.currentTimeMillis());
        
        // Example: Send data to server, process files, etc.
        // Note: For network operations, consider using WorkManager instead
    }
}

// Start service
Intent serviceIntent = new Intent(this, MyService.class);
startService(serviceIntent);

// Stop service
Intent serviceIntent = new Intent(this, MyService.class);
stopService(serviceIntent);
```

### Bound Service
```java
public class BoundService extends Service {
    
    private final IBinder binder = new LocalBinder();
    private boolean isWorking = false;
    
    public class LocalBinder extends Binder {
        public BoundService getService() {
            return BoundService.this;
        }
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        Log.d("BoundService", "Service bound");
        return binder;
    }
    
    @Override
    public boolean onUnbind(Intent intent) {
        Log.d("BoundService", "Service unbound");
        return super.onUnbind(intent);
    }
    
    // Public methods for clients
    public void startWork() {
        if (!isWorking) {
            isWorking = true;
            Log.d("BoundService", "Work started");
            // Start background work
        }
    }
    
    public void stopWork() {
        if (isWorking) {
            isWorking = false;
            Log.d("BoundService", "Work stopped");
            // Stop background work
        }
    }
    
    public boolean isWorking() {
        return isWorking;
    }
    
    public String getStatus() {
        return isWorking ? "Working" : "Idle";
    }
}

// Activity binding to service
public class MainActivity extends AppCompatActivity {
    
    private BoundService boundService;
    private boolean isServiceBound = false;
    
    private ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            BoundService.LocalBinder binder = (BoundService.LocalBinder) service;
            boundService = binder.getService();
            isServiceBound = true;
            Log.d("MainActivity", "Service connected");
        }
        
        @Override
        public void onServiceDisconnected(ComponentName name) {
            isServiceBound = false;
            Log.d("MainActivity", "Service disconnected");
        }
    };
    
    @Override
    protected void onStart() {
        super.onStart();
        // Bind to service
        Intent intent = new Intent(this, BoundService.class);
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE);
    }
    
    @Override
    protected void onStop() {
        super.onStop();
        // Unbind from service
        if (isServiceBound) {
            unbindService(serviceConnection);
            isServiceBound = false;
        }
    }
    
    public void startServiceWork(View view) {
        if (isServiceBound) {
            boundService.startWork();
        }
    }
    
    public void stopServiceWork(View view) {
        if (isServiceBound) {
            boundService.stopWork();
        }
    }
    
    public void checkServiceStatus(View view) {
        if (isServiceBound) {
            String status = boundService.getStatus();
            Toast.makeText(this, "Service status: " + status, Toast.LENGTH_SHORT).show();
        }
    }
}
```

## IntentService

### Basic IntentService
```java
public class FileProcessingService extends IntentService {
    
    public static final String ACTION_PROCESS_FILE = "com.example.ACTION_PROCESS_FILE";
    public static final String EXTRA_FILE_PATH = "file_path";
    
    public FileProcessingService() {
        super("FileProcessingService");
    }
    
    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        if (intent != null) {
            String action = intent.getAction();
            
            if (ACTION_PROCESS_FILE.equals(action)) {
                String filePath = intent.getStringExtra(EXTRA_FILE_PATH);
                processFile(filePath);
            }
        }
    }
    
    private void processFile(String filePath) {
        Log.d("FileProcessingService", "Processing file: " + filePath);
        
        try {
            // Simulate file processing
            Thread.sleep(3000);
            
            // Send broadcast when processing is complete
            Intent broadcastIntent = new Intent("com.example.FILE_PROCESSED");
            broadcastIntent.putExtra("file_path", filePath);
            broadcastIntent.putExtra("success", true);
            sendBroadcast(broadcastIntent);
            
            Log.d("FileProcessingService", "File processing completed: " + filePath);
            
        } catch (InterruptedException e) {
            Log.e("FileProcessingService", "File processing interrupted", e);
            
            // Send error broadcast
            Intent broadcastIntent = new Intent("com.example.FILE_PROCESSED");
            broadcastIntent.putExtra("file_path", filePath);
            broadcastIntent.putExtra("success", false);
            broadcastIntent.putExtra("error", e.getMessage());
            sendBroadcast(broadcastIntent);
        }
    }
}

// Start IntentService
Intent serviceIntent = new Intent(this, FileProcessingService.class);
serviceIntent.setAction(FileProcessingService.ACTION_PROCESS_FILE);
serviceIntent.putExtra(FileProcessingService.EXTRA_FILE_PATH, "/path/to/file");
startService(serviceIntent);

// Register broadcast receiver to get results
BroadcastReceiver fileProcessedReceiver = new BroadcastReceiver() {
    @Override
    public void onReceive(Context context, Intent intent) {
        String filePath = intent.getStringExtra("file_path");
        boolean success = intent.getBooleanExtra("success", false);
        
        if (success) {
            Toast.makeText(context, "File processed: " + filePath, Toast.LENGTH_SHORT).show();
        } else {
            String error = intent.getStringExtra("error");
            Toast.makeText(context, "Error processing file: " + error, Toast.LENGTH_SHORT).show();
        }
    }
};

IntentFilter filter = new IntentFilter("com.example.FILE_PROCESSED");
registerReceiver(fileProcessedReceiver, filter);
```

## JobScheduler

### JobService Implementation
```java
@TargetApi(Build.VERSION_CODES.LOLLIPOP)
public class DataSyncJobService extends JobService {
    
    private static final int JOB_ID = 1000;
    private boolean isJobRunning = false;
    
    @Override
    public boolean onStartJob(JobParameters params) {
        Log.d("DataSyncJobService", "Job started");
        
        // Job should be executed on background thread
        isJobRunning = true;
        performBackgroundWork(params);
        
        // Return true if job continues on background thread
        return true;
    }
    
    @Override
    public boolean onStopJob(JobParameters params) {
        Log.d("DataSyncJobService", "Job stopped");
        isJobRunning = false;
        
        // Return true to reschedule job
        return true;
    }
    
    private void performBackgroundWork(JobParameters params) {
        Thread thread = new Thread(() -> {
            try {
                // Simulate background work
                for (int i = 0; i < 10 && isJobRunning; i++) {
                    Thread.sleep(1000);
                    Log.d("DataSyncJobService", "Work progress: " + (i + 1));
                }
                
                if (isJobRunning) {
                    Log.d("DataSyncJobService", "Job completed successfully");
                    
                    // Job finished successfully
                    jobFinished(params, false);
                }
                
            } catch (InterruptedException e) {
                Log.e("DataSyncJobService", "Job interrupted", e);
                
                // Job failed, reschedule
                jobFinished(params, true);
            }
        });
        
        thread.start();
    }
    
    // Helper method to schedule job
    public static void scheduleJob(Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            JobScheduler jobScheduler = (JobScheduler) context.getSystemService(Context.JOB_SCHEDULER_SERVICE);
            
            JobInfo jobInfo = new JobInfo.Builder(JOB_ID, new ComponentName(context, DataSyncJobService.class))
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_CONNECTED)
                .setRequiresCharging(false)
                .setRequiresDeviceIdle(false)
                .setPersisted(true)
                .setPeriodic(15 * 60 * 1000) // 15 minutes
                .build();
            
            int result = jobScheduler.schedule(jobInfo);
            
            if (result == JobScheduler.RESULT_SUCCESS) {
                Log.d("DataSyncJobService", "Job scheduled successfully");
            } else {
                Log.e("DataSyncJobService", "Job scheduling failed");
            }
        }
    }
    
    public static void cancelJob(Context context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            JobScheduler jobScheduler = (JobScheduler) context.getSystemService(Context.JOB_SCHEDULER_SERVICE);
            jobScheduler.cancel(JOB_ID);
            Log.d("DataSyncJobService", "Job cancelled");
        }
    }
}
```

## WorkManager

### Add Dependency
```gradle
// app/build.gradle
dependencies {
    implementation "androidx.work:work-runtime:2.8.1"
}
```

### Worker Implementation
```java
public class DataSyncWorker extends Worker {
    
    public static final String WORK_NAME = "DataSyncWork";
    public static final String KEY_INPUT_DATA = "input_data";
    public static final String KEY_RESULT_DATA = "result_data";
    
    public DataSyncWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }
    
    @NonNull
    @Override
    public Result doWork() {
        Log.d("DataSyncWorker", "Work started");
        
        try {
            // Get input data
            String inputData = getInputData().getString(KEY_INPUT_DATA);
            Log.d("DataSyncWorker", "Input data: " + inputData);
            
            // Perform background work
            String result = performDataSync(inputData);
            
            // Set output data
            Data outputData = new Data.Builder()
                .putString(KEY_RESULT_DATA, result)
                .build();
            
            Log.d("DataSyncWorker", "Work completed successfully");
            return Result.success(outputData);
            
        } catch (Exception e) {
            Log.e("DataSyncWorker", "Work failed", e);
            
            // Return retry to automatically retry the work
            return Result.retry();
        }
    }
    
    private String performDataSync(String inputData) throws InterruptedException {
        // Simulate data synchronization
        for (int i = 0; i < 5; i++) {
            Thread.sleep(1000);
            Log.d("DataSyncWorker", "Sync progress: " + (i + 1) + "/5");
        }
        
        return "Sync completed for: " + inputData;
    }
    
    // Helper methods for scheduling work
    public static void scheduleOneTimeWork(Context context, String inputData) {
        Data inputDataObj = new Data.Builder()
            .putString(KEY_INPUT_DATA, inputData)
            .build();
        
        OneTimeWorkRequest workRequest = new OneTimeWorkRequest.Builder(DataSyncWorker.class)
            .setInputData(inputDataObj)
            .setConstraints(getWorkConstraints())
            .setBackoffCriteria(BackoffPolicy.LINEAR, 30, TimeUnit.SECONDS)
            .build();
        
        WorkManager.getInstance(context).enqueue(workRequest);
        
        Log.d("DataSyncWorker", "One-time work scheduled");
    }
    
    public static void schedulePeriodicWork(Context context) {
        PeriodicWorkRequest workRequest = new PeriodicWorkRequest.Builder(
            DataSyncWorker.class, 15, TimeUnit.MINUTES)
            .setConstraints(getWorkConstraints())
            .build();
        
        WorkManager.getInstance(context).enqueueUniquePeriodicWork(
            WORK_NAME,
            ExistingPeriodicWorkPolicy.KEEP,
            workRequest
        );
        
        Log.d("DataSyncWorker", "Periodic work scheduled");
    }
    
    private static Constraints getWorkConstraints() {
        return new Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .setRequiresBatteryNotLow(true)
            .build();
    }
    
    public static void cancelWork(Context context) {
        WorkManager.getInstance(context).cancelUniqueWork(WORK_NAME);
        Log.d("DataSyncWorker", "Work cancelled");
    }
}
```

### Observing Work Status
```java
public class WorkManagerActivity extends AppCompatActivity {
    
    private TextView statusTextView;
    private Button startButton, cancelButton;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_work_manager);
        
        statusTextView = findViewById(R.id.statusTextView);
        startButton = findViewById(R.id.startButton);
        cancelButton = findViewById(R.id.cancelButton);
        
        startButton.setOnClickListener(v -> startWork());
        cancelButton.setOnClickListener(v -> cancelWork());
        
        observeWorkStatus();
    }
    
    private void startWork() {
        DataSyncWorker.scheduleOneTimeWork(this, "Sample data");
        
        // Also schedule periodic work
        DataSyncWorker.schedulePeriodicWork(this);
    }
    
    private void cancelWork() {
        DataSyncWorker.cancelWork(this);
    }
    
    private void observeWorkStatus() {
        WorkManager.getInstance(this)
            .getWorkInfosForUniqueWorkLiveData(DataSyncWorker.WORK_NAME)
            .observe(this, workInfos -> {
                if (workInfos != null && !workInfos.isEmpty()) {
                    WorkInfo workInfo = workInfos.get(0);
                    
                    String status = "Status: " + workInfo.getState().name();
                    
                    if (workInfo.getState() == WorkInfo.State.SUCCEEDED) {
                        Data outputData = workInfo.getOutputData();
                        String result = outputData.getString(DataSyncWorker.KEY_RESULT_DATA);
                        status += "\nResult: " + result;
                    } else if (workInfo.getState() == WorkInfo.State.FAILED) {
                        status += "\nWork failed";
                    }
                    
                    statusTextView.setText(status);
                }
            });
    }
}
```

## Foreground Services

### Foreground Service Implementation
```java
public class DownloadService extends Service {
    
    private static final int NOTIFICATION_ID = 1;
    private static final String CHANNEL_ID = "DownloadChannel";
    
    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent.getAction();
        
        if ("START_DOWNLOAD".equals(action)) {
            String url = intent.getStringExtra("url");
            startForegroundService();
            startDownload(url);
        } else if ("STOP_DOWNLOAD".equals(action)) {
            stopSelf();
        }
        
        return START_NOT_STICKY;
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
    
    private void startForegroundService() {
        Notification notification = createNotification("Download starting...", 0);
        startForeground(NOTIFICATION_ID, notification);
    }
    
    private void startDownload(String url) {
        Thread downloadThread = new Thread(() -> {
            try {
                // Simulate download with progress updates
                for (int progress = 0; progress <= 100; progress += 10) {
                    Thread.sleep(1000);
                    updateNotification("Downloading...", progress);
                }
                
                updateNotification("Download completed", 100);
                
                // Stop foreground service after completion
                new Handler(Looper.getMainLooper()).postDelayed(() -> {
                    stopForeground(true);
                    stopSelf();
                }, 2000);
                
            } catch (InterruptedException e) {
                updateNotification("Download cancelled", 0);
                stopForeground(true);
                stopSelf();
            }
        });
        
        downloadThread.start();
    }
    
    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Download Channel",
                NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Channel for download notifications");
            
            NotificationManager notificationManager = getSystemService(NotificationManager.class);
            notificationManager.createNotificationChannel(channel);
        }
    }
    
    private Notification createNotification(String text, int progress) {
        Intent stopIntent = new Intent(this, DownloadService.class);
        stopIntent.setAction("STOP_DOWNLOAD");
        PendingIntent stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("File Download")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_download)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .addAction(R.drawable.ic_stop, "Stop", stopPendingIntent);
        
        if (progress > 0) {
            builder.setProgress(100, progress, false);
        }
        
        return builder.build();
    }
    
    private void updateNotification(String text, int progress) {
        Notification notification = createNotification(text, progress);
        NotificationManager notificationManager = 
            (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        notificationManager.notify(NOTIFICATION_ID, notification);
    }
}

// Start foreground service
Intent serviceIntent = new Intent(this, DownloadService.class);
serviceIntent.setAction("START_DOWNLOAD");
serviceIntent.putExtra("url", "https://example.com/file.zip");

if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
    startForegroundService(serviceIntent);
} else {
    startService(serviceIntent);
}
```

### Manifest Declaration
```xml
<!-- AndroidManifest.xml -->
<service
    android:name=".DownloadService"
    android:enabled="true"
    android:exported="false"
    android:foregroundServiceType="dataSync" />

<!-- For Android 9+ -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
```

## Best Practices

### 1. Choose the Right Background Task Solution
```java
public class BackgroundTaskManager {
    
    // Use this for quick, UI-related background tasks
    public void executeQuickTask(Runnable task) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Handler mainHandler = new Handler(Looper.getMainLooper());
        
        executor.execute(() -> {
            // Background work
            task.run();
            
            // If you need to update UI, post to main thread
            mainHandler.post(() -> {
                // UI updates
            });
        });
        
        executor.shutdown();
    }
    
    // Use WorkManager for deferrable, guaranteed work
    public void scheduleWorkManagerTask() {
        OneTimeWorkRequest workRequest = new OneTimeWorkRequest.Builder(DataSyncWorker.class)
            .setConstraints(new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build())
            .build();
        
        WorkManager.getInstance(this).enqueue(workRequest);
    }
    
    // Use foreground service for user-visible, long-running tasks
    public void startForegroundTask() {
        Intent serviceIntent = new Intent(this, DownloadService.class);
        serviceIntent.setAction("START_DOWNLOAD");
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent);
        } else {
            startService(serviceIntent);
        }
    }
}
```

### 2. Handle Configuration Changes
```java
public class TaskActivity extends AppCompatActivity {
    
    private TaskExecutor taskExecutor;
    private boolean isTaskRunning = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_task);
        
        taskExecutor = new TaskExecutor();
        
        // Restore state
        if (savedInstanceState != null) {
            isTaskRunning = savedInstanceState.getBoolean("task_running", false);
        }
    }
    
    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putBoolean("task_running", isTaskRunning);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (taskExecutor != null) {
            taskExecutor.shutdown();
        }
    }
    
    public void startTask(View view) {
        if (!isTaskRunning) {
            isTaskRunning = true;
            
            taskExecutor.executeTask(() -> {
                // Long running task
                Thread.sleep(10000);
                return "Task completed";
            }, new TaskExecutor.OnTaskListener<String>() {
                @Override
                public void onSuccess(String result) {
                    isTaskRunning = false;
                    // Update UI
                }
                
                @Override
                public void onError(String error) {
                    isTaskRunning = false;
                    // Handle error
                }
            });
        }
    }
}
```

### 3. Memory Management
```java
public class MemoryEfficientTask extends AsyncTask<Void, Void, String> {
    
    private WeakReference<Activity> activityRef;
    
    public MemoryEfficientTask(Activity activity) {
        this.activityRef = new WeakReference<>(activity);
    }
    
    @Override
    protected String doInBackground(Void... voids) {
        // Background work
        return "Result";
    }
    
    @Override
    protected void onPostExecute(String result) {
        Activity activity = activityRef.get();
        if (activity != null && !activity.isFinishing()) {
            // Update UI safely
        }
    }
}
```

### 4. Error Handling
```java
public class RobustTaskExecutor {
    
    public void executeWithRetry(Callable<String> task, int maxRetries) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        
        executor.execute(() -> {
            int attempt = 0;
            Exception lastException = null;
            
            while (attempt < maxRetries) {
                try {
                    String result = task.call();
                    // Success
                    handleSuccess(result);
                    return;
                    
                } catch (Exception e) {
                    lastException = e;
                    attempt++;
                    
                    if (attempt < maxRetries) {
                        try {
                            Thread.sleep(1000 * attempt); // Exponential backoff
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            break;
                        }
                    }
                }
            }
            
            // All retries failed
            handleError(lastException);
        });
        
        executor.shutdown();
    }
    
    private void handleSuccess(String result) {
        new Handler(Looper.getMainLooper()).post(() -> {
            // Update UI with success
        });
    }
    
    private void handleError(Exception error) {
        new Handler(Looper.getMainLooper()).post(() -> {
            // Show error to user
        });
    }
}
```

Understanding background tasks and services is crucial for creating responsive Android applications. Choose the appropriate solution based on your specific needs: WorkManager for guaranteed execution, ExecutorService for simple threading, and foreground services for user-visible long-running operations.

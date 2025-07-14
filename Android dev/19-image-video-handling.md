# Image and Video Handling

## Table of Contents
- [Overview of Media Handling](#overview-of-media-handling)
- [Camera Integration](#camera-integration)
- [Image Capture and Processing](#image-capture-and-processing)
- [Gallery and File Access](#gallery-and-file-access)
- [Image Loading and Display](#image-loading-and-display)
- [Video Recording](#video-recording)
- [Video Playback](#video-playback)
- [Image Editing and Filters](#image-editing-and-filters)
- [Media Storage and Management](#media-storage-and-management)
- [Permissions and Security](#permissions-and-security)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

## Overview of Media Handling

Android provides comprehensive APIs for capturing, processing, and displaying images and videos. These capabilities enable rich multimedia experiences in applications.

### Key Components
- **Camera API**: Direct camera control and image capture
- **MediaStore**: Access to device media collections
- **Intent-based capture**: Simple image/video capture using system apps
- **ExifInterface**: Read and write image metadata
- **BitmapFactory**: Image loading and manipulation
- **MediaPlayer**: Video playback
- **MediaRecorder**: Video recording

### Permissions Required
```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />

<!-- Camera features -->
<uses-feature android:name="android.hardware.camera" android:required="true" />
<uses-feature android:name="android.hardware.camera.autofocus" />
```

## Camera Integration

### Intent-based Camera Capture
The simplest way to capture images using the system camera app.

```java
public class CameraCaptureActivity extends AppCompatActivity {
    
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_VIDEO_CAPTURE = 2;
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    
    private ImageView imageView;
    private VideoView videoView;
    private Uri photoUri;
    private Uri videoUri;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_capture);
        
        initializeViews();
        setupClickListeners();
    }
    
    private void initializeViews() {
        imageView = findViewById(R.id.imageView);
        videoView = findViewById(R.id.videoView);
        
        Button captureImageButton = findViewById(R.id.captureImageButton);
        Button captureVideoButton = findViewById(R.id.captureVideoButton);
        
        captureImageButton.setOnClickListener(v -> checkPermissionAndCaptureImage());
        captureVideoButton.setOnClickListener(v -> checkPermissionAndCaptureVideo());
    }
    
    private void checkPermissionAndCaptureImage() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            captureImage();
        }
    }
    
    private void checkPermissionAndCaptureVideo() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                != PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO}, 
                REQUEST_CAMERA_PERMISSION);
        } else {
            captureVideo();
        }
    }
    
    private void captureImage() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        
        // Create file to save the image
        File photoFile = createImageFile();
        if (photoFile != null) {
            photoUri = FileProvider.getUriForFile(this,
                "com.example.myapp.fileprovider", photoFile);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }
    
    private void captureVideo() {
        Intent takeVideoIntent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
        
        // Create file to save the video
        File videoFile = createVideoFile();
        if (videoFile != null) {
            videoUri = FileProvider.getUriForFile(this,
                "com.example.myapp.fileprovider", videoFile);
            takeVideoIntent.putExtra(MediaStore.EXTRA_OUTPUT, videoUri);
            takeVideoIntent.putExtra(MediaStore.EXTRA_DURATION_LIMIT, 30); // 30 seconds max
            takeVideoIntent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 1); // High quality
            
            if (takeVideoIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takeVideoIntent, REQUEST_VIDEO_CAPTURE);
            }
        }
    }
    
    private File createImageFile() {
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                .format(new Date());
            String imageFileName = "JPEG_" + timeStamp + "_";
            File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
            
            return File.createTempFile(imageFileName, ".jpg", storageDir);
        } catch (IOException ex) {
            Log.e("CameraCapture", "Error creating image file", ex);
            return null;
        }
    }
    
    private File createVideoFile() {
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                .format(new Date());
            String videoFileName = "MP4_" + timeStamp + "_";
            File storageDir = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
            
            return File.createTempFile(videoFileName, ".mp4", storageDir);
        } catch (IOException ex) {
            Log.e("CameraCapture", "Error creating video file", ex);
            return null;
        }
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (resultCode == RESULT_OK) {
            switch (requestCode) {
                case REQUEST_IMAGE_CAPTURE:
                    handleImageCapture();
                    break;
                case REQUEST_VIDEO_CAPTURE:
                    handleVideoCapture();
                    break;
            }
        }
    }
    
    private void handleImageCapture() {
        if (photoUri != null) {
            try {
                // Load and display the captured image
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(
                    getContentResolver(), photoUri);
                
                // Rotate if necessary based on EXIF data
                bitmap = rotateImageIfRequired(bitmap, photoUri);
                
                imageView.setImageBitmap(bitmap);
                imageView.setVisibility(View.VISIBLE);
                videoView.setVisibility(View.GONE);
                
                // Add to gallery
                addImageToGallery();
                
            } catch (IOException e) {
                Log.e("CameraCapture", "Error loading captured image", e);
            }
        }
    }
    
    private void handleVideoCapture() {
        if (videoUri != null) {
            videoView.setVideoURI(videoUri);
            videoView.setOnPreparedListener(mp -> {
                mp.setLooping(true);
                videoView.start();
            });
            
            videoView.setVisibility(View.VISIBLE);
            imageView.setVisibility(View.GONE);
            
            // Add to gallery
            addVideoToGallery();
        }
    }
    
    private Bitmap rotateImageIfRequired(Bitmap bitmap, Uri imageUri) {
        try {
            ExifInterface exif = new ExifInterface(
                getContentResolver().openInputStream(imageUri));
            int orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    return rotateImage(bitmap, 90);
                case ExifInterface.ORIENTATION_ROTATE_180:
                    return rotateImage(bitmap, 180);
                case ExifInterface.ORIENTATION_ROTATE_270:
                    return rotateImage(bitmap, 270);
                default:
                    return bitmap;
            }
        } catch (IOException e) {
            Log.e("CameraCapture", "Error reading EXIF data", e);
            return bitmap;
        }
    }
    
    private Bitmap rotateImage(Bitmap bitmap, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(bitmap, 0, 0, 
            bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
    
    private void addImageToGallery() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        mediaScanIntent.setData(photoUri);
        sendBroadcast(mediaScanIntent);
    }
    
    private void addVideoToGallery() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        mediaScanIntent.setData(videoUri);
        sendBroadcast(mediaScanIntent);
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, 
                                         int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, can proceed with camera operations
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
```

### Camera API 2 Implementation
For more control over camera operations, use Camera API 2.

```java
public class Camera2Activity extends AppCompatActivity {
    
    private TextureView textureView;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder captureRequestBuilder;
    private ImageReader imageReader;
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private String cameraId;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera2);
        
        textureView = findViewById(R.id.textureView);
        textureView.setSurfaceTextureListener(textureListener);
        
        Button captureButton = findViewById(R.id.captureButton);
        captureButton.setOnClickListener(v -> capturePhoto());
    }
    
    private TextureView.SurfaceTextureListener textureListener = 
        new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            openCamera();
        }
        
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Handle surface size changes
        }
        
        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }
        
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            // Handle frame updates
        }
    };
    
    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }
    
    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
        } catch (InterruptedException e) {
            Log.e("Camera2", "Error stopping background thread", e);
        }
    }
    
    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        
        try {
            cameraId = manager.getCameraIdList()[0]; // Use back camera
            
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, 
                    new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            
            manager.openCamera(cameraId, stateCallback, backgroundHandler);
            
        } catch (CameraAccessException e) {
            Log.e("Camera2", "Error opening camera", e);
        }
    }
    
    private CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreview();
        }
        
        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }
        
        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };
    
    private void createCameraPreview() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            texture.setDefaultBufferSize(1920, 1080);
            Surface surface = new Surface(texture);
            
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
            
            cameraDevice.createCaptureSession(Arrays.asList(surface), 
                new CameraCaptureSession.StateCallback() {
                    @Override
                    public void onConfigured(CameraCaptureSession session) {
                        if (cameraDevice == null) return;
                        
                        captureSession = session;
                        updatePreview();
                    }
                    
                    @Override
                    public void onConfigureFailed(CameraCaptureSession session) {
                        Toast.makeText(Camera2Activity.this, 
                            "Configuration failed", Toast.LENGTH_SHORT).show();
                    }
                }, null);
                
        } catch (CameraAccessException e) {
            Log.e("Camera2", "Error creating camera preview", e);
        }
    }
    
    private void updatePreview() {
        if (cameraDevice == null) return;
        
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        
        try {
            captureSession.setRepeatingRequest(captureRequestBuilder.build(), 
                null, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e("Camera2", "Error updating preview", e);
        }
    }
    
    private void capturePhoto() {
        if (cameraDevice == null) return;
        
        try {
            // Set up image reader
            imageReader = ImageReader.newInstance(1920, 1080, ImageFormat.JPEG, 1);
            
            List<Surface> outputs = Arrays.asList(imageReader.getSurface());
            
            CaptureRequest.Builder captureBuilder = 
                cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            captureBuilder.addTarget(imageReader.getSurface());
            captureBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
            
            // Set orientation
            int rotation = getWindowManager().getDefaultDisplay().getRotation();
            captureBuilder.set(CaptureRequest.JPEG_ORIENTATION, getOrientation(rotation));
            
            ImageReader.OnImageAvailableListener listener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    Image image = reader.acquireLatestImage();
                    saveImage(image);
                }
            };
            
            imageReader.setOnImageAvailableListener(listener, backgroundHandler);
            
            captureSession.capture(captureBuilder.build(), null, backgroundHandler);
            
        } catch (CameraAccessException e) {
            Log.e("Camera2", "Error capturing photo", e);
        }
    }
    
    private void saveImage(Image image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        
        File file = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), 
            "IMG_" + System.currentTimeMillis() + ".jpg");
        
        try (FileOutputStream output = new FileOutputStream(file)) {
            output.write(bytes);
            
            runOnUiThread(() -> 
                Toast.makeText(this, "Photo saved: " + file.getAbsolutePath(), 
                    Toast.LENGTH_SHORT).show()
            );
            
        } catch (IOException e) {
            Log.e("Camera2", "Error saving image", e);
        } finally {
            image.close();
        }
    }
    
    private int getOrientation(int rotation) {
        switch (rotation) {
            case Surface.ROTATION_0:
                return 90;
            case Surface.ROTATION_90:
                return 0;
            case Surface.ROTATION_180:
                return 270;
            case Surface.ROTATION_270:
                return 180;
        }
        return 0;
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }
    
    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }
    
    private void closeCamera() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
        
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
    }
}
```

## Image Capture and Processing

### Image Processing and Manipulation
```java
public class ImageProcessingUtil {
    
    public static Bitmap resizeBitmap(Bitmap original, int maxWidth, int maxHeight) {
        int width = original.getWidth();
        int height = original.getHeight();
        
        float bitmapRatio = (float) width / (float) height;
        
        if (bitmapRatio > 1) {
            // Landscape
            width = maxWidth;
            height = (int) (width / bitmapRatio);
        } else {
            // Portrait
            height = maxHeight;
            width = (int) (height * bitmapRatio);
        }
        
        return Bitmap.createScaledBitmap(original, width, height, true);
    }
    
    public static Bitmap cropBitmapToSquare(Bitmap bitmap) {
        int size = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int x = (bitmap.getWidth() - size) / 2;
        int y = (bitmap.getHeight() - size) / 2;
        
        return Bitmap.createBitmap(bitmap, x, y, size, size);
    }
    
    public static Bitmap applyGrayscaleFilter(Bitmap original) {
        Bitmap grayscale = Bitmap.createBitmap(
            original.getWidth(), original.getHeight(), Bitmap.Config.ARGB_8888);
        
        Canvas canvas = new Canvas(grayscale);
        Paint paint = new Paint();
        
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0);
        
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(colorMatrix);
        paint.setColorFilter(filter);
        
        canvas.drawBitmap(original, 0, 0, paint);
        
        return grayscale;
    }
    
    public static Bitmap applySepiaFilter(Bitmap original) {
        Bitmap sepia = Bitmap.createBitmap(
            original.getWidth(), original.getHeight(), Bitmap.Config.ARGB_8888);
        
        Canvas canvas = new Canvas(sepia);
        Paint paint = new Paint();
        
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.set(new float[]{
            0.393f, 0.769f, 0.189f, 0, 0,
            0.349f, 0.686f, 0.168f, 0, 0,
            0.272f, 0.534f, 0.131f, 0, 0,
            0, 0, 0, 1, 0
        });
        
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(colorMatrix);
        paint.setColorFilter(filter);
        
        canvas.drawBitmap(original, 0, 0, paint);
        
        return sepia;
    }
    
    public static Bitmap addWatermark(Bitmap original, String watermarkText) {
        Bitmap watermarked = original.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(watermarked);
        
        Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setTextSize(50f);
        paint.setAntiAlias(true);
        paint.setTypeface(Typeface.DEFAULT_BOLD);
        paint.setShadowLayer(2f, 2f, 2f, Color.BLACK);
        
        float x = 20;
        float y = watermarked.getHeight() - 20;
        
        canvas.drawText(watermarkText, x, y, paint);
        
        return watermarked;
    }
    
    public static void saveImageToFile(Bitmap bitmap, File file, int quality) throws IOException {
        try (FileOutputStream out = new FileOutputStream(file)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, out);
        }
    }
    
    public static Bitmap loadOptimizedBitmap(String imagePath, int reqWidth, int reqHeight) {
        // First decode with inJustDecodeBounds=true to check dimensions
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(imagePath, options);
        
        // Calculate inSampleSize
        options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);
        
        // Decode bitmap with inSampleSize set
        options.inJustDecodeBounds = false;
        return BitmapFactory.decodeFile(imagePath, options);
    }
    
    private static int calculateInSampleSize(BitmapFactory.Options options, 
                                           int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;
        
        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;
            
            while ((halfHeight / inSampleSize) >= reqHeight
                    && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        
        return inSampleSize;
    }
}
```

## Gallery and File Access

### Accessing Device Media
```java
public class GalleryAccessActivity extends AppCompatActivity {
    
    private static final int REQUEST_PICK_IMAGE = 1;
    private static final int REQUEST_PICK_VIDEO = 2;
    private static final int REQUEST_STORAGE_PERMISSION = 100;
    
    private RecyclerView recyclerView;
    private MediaAdapter mediaAdapter;
    private List<MediaItem> mediaItems;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery_access);
        
        initializeViews();
        checkPermissionAndLoadMedia();
    }
    
    private void initializeViews() {
        recyclerView = findViewById(R.id.recyclerView);
        recyclerView.setLayoutManager(new GridLayoutManager(this, 3));
        
        mediaItems = new ArrayList<>();
        mediaAdapter = new MediaAdapter(mediaItems, this::onMediaItemClick);
        recyclerView.setAdapter(mediaAdapter);
        
        Button pickImageButton = findViewById(R.id.pickImageButton);
        Button pickVideoButton = findViewById(R.id.pickVideoButton);
        
        pickImageButton.setOnClickListener(v -> pickImageFromGallery());
        pickVideoButton.setOnClickListener(v -> pickVideoFromGallery());
    }
    
    private void checkPermissionAndLoadMedia() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 
                REQUEST_STORAGE_PERMISSION);
        } else {
            loadMediaFromGallery();
        }
    }
    
    private void loadMediaFromGallery() {
        new AsyncTask<Void, Void, List<MediaItem>>() {
            @Override
            protected List<MediaItem> doInBackground(Void... voids) {
                List<MediaItem> items = new ArrayList<>();
                
                // Load images
                items.addAll(loadImages());
                
                // Load videos
                items.addAll(loadVideos());
                
                // Sort by date (newest first)
                Collections.sort(items, 
                    (a, b) -> Long.compare(b.dateAdded, a.dateAdded));
                
                return items;
            }
            
            @Override
            protected void onPostExecute(List<MediaItem> items) {
                mediaItems.clear();
                mediaItems.addAll(items);
                mediaAdapter.notifyDataSetChanged();
            }
        }.execute();
    }
    
    private List<MediaItem> loadImages() {
        List<MediaItem> images = new ArrayList<>();
        
        String[] projection = {
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATA,
            MediaStore.Images.Media.DATE_ADDED
        };
        
        Cursor cursor = getContentResolver().query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            MediaStore.Images.Media.DATE_ADDED + " DESC"
        );
        
        if (cursor != null) {
            int idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID);
            int nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME);
            int dataColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            int dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED);
            
            while (cursor.moveToNext()) {
                long id = cursor.getLong(idColumn);
                String name = cursor.getString(nameColumn);
                String path = cursor.getString(dataColumn);
                long dateAdded = cursor.getLong(dateColumn);
                
                Uri uri = ContentUris.withAppendedId(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id);
                
                images.add(new MediaItem(id, name, path, uri, MediaItem.Type.IMAGE, dateAdded));
            }
            
            cursor.close();
        }
        
        return images;
    }
    
    private List<MediaItem> loadVideos() {
        List<MediaItem> videos = new ArrayList<>();
        
        String[] projection = {
            MediaStore.Video.Media._ID,
            MediaStore.Video.Media.DISPLAY_NAME,
            MediaStore.Video.Media.DATA,
            MediaStore.Video.Media.DATE_ADDED
        };
        
        Cursor cursor = getContentResolver().query(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            MediaStore.Video.Media.DATE_ADDED + " DESC"
        );
        
        if (cursor != null) {
            int idColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media._ID);
            int nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DISPLAY_NAME);
            int dataColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATA);
            int dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATE_ADDED);
            
            while (cursor.moveToNext()) {
                long id = cursor.getLong(idColumn);
                String name = cursor.getString(nameColumn);
                String path = cursor.getString(dataColumn);
                long dateAdded = cursor.getLong(dateColumn);
                
                Uri uri = ContentUris.withAppendedId(
                    MediaStore.Video.Media.EXTERNAL_CONTENT_URI, id);
                
                videos.add(new MediaItem(id, name, path, uri, MediaItem.Type.VIDEO, dateAdded));
            }
            
            cursor.close();
        }
        
        return videos;
    }
    
    private void pickImageFromGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, 
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_PICK_IMAGE);
    }
    
    private void pickVideoFromGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, 
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_PICK_VIDEO);
    }
    
    private void onMediaItemClick(MediaItem item) {
        if (item.type == MediaItem.Type.IMAGE) {
            // Open image viewer
            Intent intent = new Intent(this, ImageViewerActivity.class);
            intent.putExtra("image_uri", item.uri.toString());
            startActivity(intent);
        } else {
            // Open video player
            Intent intent = new Intent(this, VideoPlayerActivity.class);
            intent.putExtra("video_uri", item.uri.toString());
            startActivity(intent);
        }
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (resultCode == RESULT_OK && data != null) {
            Uri selectedMediaUri = data.getData();
            
            switch (requestCode) {
                case REQUEST_PICK_IMAGE:
                    handleSelectedImage(selectedMediaUri);
                    break;
                case REQUEST_PICK_VIDEO:
                    handleSelectedVideo(selectedMediaUri);
                    break;
            }
        }
    }
    
    private void handleSelectedImage(Uri imageUri) {
        // Process selected image
        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
            // Use the bitmap as needed
            Toast.makeText(this, "Image selected", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Log.e("Gallery", "Error loading selected image", e);
        }
    }
    
    private void handleSelectedVideo(Uri videoUri) {
        // Process selected video
        Toast.makeText(this, "Video selected: " + videoUri.toString(), Toast.LENGTH_SHORT).show();
    }
    
    // MediaItem class
    private static class MediaItem {
        enum Type { IMAGE, VIDEO }
        
        long id;
        String name;
        String path;
        Uri uri;
        Type type;
        long dateAdded;
        
        MediaItem(long id, String name, String path, Uri uri, Type type, long dateAdded) {
            this.id = id;
            this.name = name;
            this.path = path;
            this.uri = uri;
            this.type = type;
            this.dateAdded = dateAdded;
        }
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, 
                                         int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == REQUEST_STORAGE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                loadMediaFromGallery();
            } else {
                Toast.makeText(this, "Storage permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
```

## Image Loading and Display

### Efficient Image Loading with Glide
```java
public class ImageDisplayActivity extends AppCompatActivity {
    
    private ImageView imageView;
    private ProgressBar progressBar;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_display);
        
        imageView = findViewById(R.id.imageView);
        progressBar = findViewById(R.id.progressBar);
        
        setupImageLoading();
    }
    
    private void setupImageLoading() {
        String imageUrl = "https://example.com/image.jpg";
        
        // Basic Glide usage
        Glide.with(this)
            .load(imageUrl)
            .into(imageView);
        
        // Advanced Glide usage with options
        loadImageWithOptions(imageUrl);
    }
    
    private void loadImageWithOptions(String imageUrl) {
        RequestOptions options = new RequestOptions()
            .placeholder(R.drawable.placeholder_image)
            .error(R.drawable.error_image)
            .centerCrop()
            .diskCacheStrategy(DiskCacheStrategy.ALL)
            .timeout(10000);
        
        Glide.with(this)
            .load(imageUrl)
            .apply(options)
            .listener(new RequestListener<Drawable>() {
                @Override
                public boolean onLoadFailed(@Nullable GlideException e, Object model,
                                          Target<Drawable> target, boolean isFirstResource) {
                    progressBar.setVisibility(View.GONE);
                    Toast.makeText(ImageDisplayActivity.this, 
                        "Failed to load image", Toast.LENGTH_SHORT).show();
                    return false;
                }
                
                @Override
                public boolean onResourceReady(Drawable resource, Object model,
                                             Target<Drawable> target, DataSource dataSource,
                                             boolean isFirstResource) {
                    progressBar.setVisibility(View.GONE);
                    return false;
                }
            })
            .into(imageView);
    }
    
    private void loadCircularImage(String imageUrl, ImageView targetView) {
        Glide.with(this)
            .load(imageUrl)
            .apply(RequestOptions.circleCropTransform())
            .into(targetView);
    }
    
    private void loadRoundedCornerImage(String imageUrl, ImageView targetView) {
        Glide.with(this)
            .load(imageUrl)
            .apply(RequestOptions.bitmapTransform(new RoundedCorners(20)))
            .into(targetView);
    }
    
    private void loadImageWithCustomTransformation(String imageUrl, ImageView targetView) {
        Glide.with(this)
            .load(imageUrl)
            .apply(RequestOptions.bitmapTransform(new BlurTransformation(25, 3)))
            .into(targetView);
    }
}

// Custom Blur Transformation
public class BlurTransformation extends BitmapTransformation {
    
    private int radius;
    private int sampling;
    
    public BlurTransformation(int radius, int sampling) {
        this.radius = radius;
        this.sampling = sampling;
    }
    
    @Override
    protected Bitmap transform(@NonNull BitmapPool pool, @NonNull Bitmap toTransform, 
                             int outWidth, int outHeight) {
        int width = toTransform.getWidth();
        int height = toTransform.getHeight();
        
        Bitmap bitmap = pool.get(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        
        Paint paint = new Paint();
        paint.setFlags(Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(toTransform, 0, 0, paint);
        
        return blur(pool, bitmap);
    }
    
    private Bitmap blur(BitmapPool pool, Bitmap bitmap) {
        // Implement blur algorithm or use RenderScript
        // This is a simplified implementation
        return bitmap;
    }
    
    @Override
    public void updateDiskCacheKey(@NonNull MessageDigest messageDigest) {
        messageDigest.update(("blur" + radius + sampling).getBytes(CHARSET));
    }
}
```

## Video Recording

### Custom Video Recording
```java
public class VideoRecordingActivity extends AppCompatActivity {
    
    private TextureView textureView;
    private MediaRecorder mediaRecorder;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    
    private boolean isRecording = false;
    private String videoFilePath;
    private Button recordButton;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video_recording);
        
        textureView = findViewById(R.id.textureView);
        recordButton = findViewById(R.id.recordButton);
        
        recordButton.setOnClickListener(v -> toggleRecording());
    }
    
    private void toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }
    
    private void startRecording() {
        try {
            setupMediaRecorder();
            
            SurfaceTexture texture = textureView.getSurfaceTexture();
            texture.setDefaultBufferSize(1920, 1080);
            Surface previewSurface = new Surface(texture);
            Surface recordSurface = mediaRecorder.getSurface();
            
            CaptureRequest.Builder captureBuilder = 
                cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            captureBuilder.addTarget(previewSurface);
            captureBuilder.addTarget(recordSurface);
            
            cameraDevice.createCaptureSession(
                Arrays.asList(previewSurface, recordSurface),
                new CameraCaptureSession.StateCallback() {
                    @Override
                    public void onConfigured(CameraCaptureSession session) {
                        captureSession = session;
                        
                        try {
                            captureSession.setRepeatingRequest(
                                captureBuilder.build(), null, backgroundHandler);
                            
                            mediaRecorder.start();
                            isRecording = true;
                            
                            runOnUiThread(() -> {
                                recordButton.setText("Stop Recording");
                                recordButton.setBackgroundColor(Color.RED);
                            });
                            
                        } catch (CameraAccessException e) {
                            Log.e("VideoRecording", "Error starting recording", e);
                        }
                    }
                    
                    @Override
                    public void onConfigureFailed(CameraCaptureSession session) {
                        Toast.makeText(VideoRecordingActivity.this, 
                            "Failed to configure camera", Toast.LENGTH_SHORT).show();
                    }
                }, backgroundHandler);
                
        } catch (Exception e) {
            Log.e("VideoRecording", "Error setting up recording", e);
        }
    }
    
    private void stopRecording() {
        try {
            mediaRecorder.stop();
            mediaRecorder.reset();
            
            isRecording = false;
            
            runOnUiThread(() -> {
                recordButton.setText("Start Recording");
                recordButton.setBackgroundColor(Color.GREEN);
                
                Toast.makeText(this, "Video saved: " + videoFilePath, 
                    Toast.LENGTH_SHORT).show();
            });
            
            // Add to gallery
            addVideoToGallery();
            
        } catch (Exception e) {
            Log.e("VideoRecording", "Error stopping recording", e);
        }
    }
    
    private void setupMediaRecorder() throws IOException {
        mediaRecorder = new MediaRecorder();
        
        // Create output file
        File videoFile = createVideoFile();
        videoFilePath = videoFile.getAbsolutePath();
        
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setOutputFile(videoFilePath);
        mediaRecorder.setVideoEncodingBitRate(10000000);
        mediaRecorder.setVideoFrameRate(30);
        mediaRecorder.setVideoSize(1920, 1080);
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        
        mediaRecorder.prepare();
    }
    
    private File createVideoFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            .format(new Date());
        String videoFileName = "VID_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
        
        return File.createTempFile(videoFileName, ".mp4", storageDir);
    }
    
    private void addVideoToGallery() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File videoFile = new File(videoFilePath);
        Uri contentUri = Uri.fromFile(videoFile);
        mediaScanIntent.setData(contentUri);
        sendBroadcast(mediaScanIntent);
    }
}
```

## Video Playback

### Custom Video Player
```java
public class VideoPlayerActivity extends AppCompatActivity {
    
    private VideoView videoView;
    private MediaController mediaController;
    private ProgressBar progressBar;
    private TextView statusText;
    private Uri videoUri;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video_player);
        
        initializeViews();
        
        String videoUriString = getIntent().getStringExtra("video_uri");
        if (videoUriString != null) {
            videoUri = Uri.parse(videoUriString);
            setupVideoPlayer();
        }
    }
    
    private void initializeViews() {
        videoView = findViewById(R.id.videoView);
        progressBar = findViewById(R.id.progressBar);
        statusText = findViewById(R.id.statusText);
        
        // Create media controller
        mediaController = new MediaController(this);
        mediaController.setAnchorView(videoView);
        videoView.setMediaController(mediaController);
    }
    
    private void setupVideoPlayer() {
        videoView.setVideoURI(videoUri);
        
        videoView.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                progressBar.setVisibility(View.GONE);
                statusText.setText("Ready to play");
                
                // Auto-start playback
                videoView.start();
                
                // Get video info
                int duration = mp.getDuration();
                int width = mp.getVideoWidth();
                int height = mp.getVideoHeight();
                
                String info = String.format(Locale.getDefault(),
                    "Duration: %d seconds, Size: %dx%d", 
                    duration / 1000, width, height);
                statusText.setText(info);
            }
        });
        
        videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                statusText.setText("Playback completed");
                finish();
            }
        });
        
        videoView.setOnErrorListener(new MediaPlayer.OnErrorListener() {
            @Override
            public boolean onError(MediaPlayer mp, int what, int extra) {
                progressBar.setVisibility(View.GONE);
                statusText.setText("Error playing video");
                
                String errorMessage = getErrorMessage(what, extra);
                Toast.makeText(VideoPlayerActivity.this, errorMessage, Toast.LENGTH_LONG).show();
                
                return true;
            }
        });
        
        progressBar.setVisibility(View.VISIBLE);
        statusText.setText("Loading video...");
    }
    
    private String getErrorMessage(int what, int extra) {
        String errorMessage = "Unknown error";
        
        switch (what) {
            case MediaPlayer.MEDIA_ERROR_UNKNOWN:
                errorMessage = "Unknown media error";
                break;
            case MediaPlayer.MEDIA_ERROR_SERVER_DIED:
                errorMessage = "Media server died";
                break;
        }
        
        switch (extra) {
            case MediaPlayer.MEDIA_ERROR_IO:
                errorMessage += " (IO error)";
                break;
            case MediaPlayer.MEDIA_ERROR_MALFORMED:
                errorMessage += " (Malformed media)";
                break;
            case MediaPlayer.MEDIA_ERROR_UNSUPPORTED:
                errorMessage += " (Unsupported media)";
                break;
            case MediaPlayer.MEDIA_ERROR_TIMED_OUT:
                errorMessage += " (Timed out)";
                break;
        }
        
        return errorMessage;
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        if (videoView.isPlaying()) {
            videoView.pause();
        }
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        if (videoView != null) {
            videoView.resume();
        }
    }
}
```

## Best Practices

### Image and Video Optimization
- **Use appropriate image formats**: JPEG for photos, PNG for graphics with transparency
- **Optimize image sizes**: Scale images to required dimensions before displaying
- **Implement proper caching**: Use libraries like Glide for efficient image caching
- **Handle memory properly**: Recycle bitmaps and avoid memory leaks
- **Use background threads**: Perform image processing off the main thread

### Security Considerations
- **Validate file types**: Check file extensions and MIME types
- **Scan for malicious content**: Use virus scanning for user-uploaded content
- **Respect privacy**: Handle camera and storage permissions appropriately
- **Secure file storage**: Store sensitive media in app-private directories

### Performance Tips
- **Use image compression**: Reduce file sizes without significant quality loss
- **Implement lazy loading**: Load images on-demand in lists and grids
- **Optimize video quality**: Balance quality and file size for video recording
- **Handle device limitations**: Adapt to different camera capabilities and storage space

Understanding image and video handling in Android enables creating rich multimedia applications with professional-quality media capture and playback capabilities.

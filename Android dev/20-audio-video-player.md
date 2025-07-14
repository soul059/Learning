# Audio and Video Player

## Table of Contents
- [Overview of Media Playback](#overview-of-media-playback)
- [Audio Playback with MediaPlayer](#audio-playback-with-mediaplayer)
- [Advanced Audio Features](#advanced-audio-features)
- [Video Playback Implementation](#video-playback-implementation)
- [Custom Media Player Controls](#custom-media-player-controls)
- [Audio Recording](#audio-recording)
- [Media Session and Notification](#media-session-and-notification)
- [Background Audio Playback](#background-audio-playback)
- [Streaming Media](#streaming-media)
- [Audio Focus Management](#audio-focus-management)
- [Performance and Optimization](#performance-and-optimization)
- [Best Practices](#best-practices)

## Overview of Media Playback

Android provides several APIs for audio and video playback, each suited for different use cases and requirements.

### Media Playback APIs
- **MediaPlayer**: Simple audio and video playback
- **VideoView**: Easy video playback with built-in controls
- **ExoPlayer**: Advanced media player with streaming support
- **SoundPool**: Low-latency audio playback for games
- **AudioTrack**: Low-level audio output
- **MediaRecorder**: Audio and video recording

### Supported Media Formats
#### Audio Formats
- MP3, AAC, FLAC, OGG Vorbis, WAV, AMR

#### Video Formats
- MP4, 3GP, WebM, MKV

### Required Permissions
```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

## Audio Playback with MediaPlayer

### Basic Audio Player Implementation
```java
public class AudioPlayerActivity extends AppCompatActivity {
    
    private MediaPlayer mediaPlayer;
    private Button playButton, pauseButton, stopButton;
    private SeekBar seekBar;
    private TextView currentTimeText, durationText;
    private Handler handler;
    private Runnable updateSeekBar;
    
    private boolean isPlaying = false;
    private boolean isPrepared = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_audio_player);
        
        initializeViews();
        setupMediaPlayer();
        setupSeekBarUpdater();
    }
    
    private void initializeViews() {
        playButton = findViewById(R.id.playButton);
        pauseButton = findViewById(R.id.pauseButton);
        stopButton = findViewById(R.id.stopButton);
        seekBar = findViewById(R.id.seekBar);
        currentTimeText = findViewById(R.id.currentTimeText);
        durationText = findViewById(R.id.durationText);
        
        playButton.setOnClickListener(v -> playAudio());
        pauseButton.setOnClickListener(v -> pauseAudio());
        stopButton.setOnClickListener(v -> stopAudio());
        
        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser && isPrepared) {
                    mediaPlayer.seekTo(progress);
                    updateCurrentTime(progress);
                }
            }
            
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                handler.removeCallbacks(updateSeekBar);
            }
            
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                if (isPlaying) {
                    handler.post(updateSeekBar);
                }
            }
        });
        
        handler = new Handler(Looper.getMainLooper());
    }
    
    private void setupMediaPlayer() {
        mediaPlayer = new MediaPlayer();
        
        // Set up listeners
        mediaPlayer.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                isPrepared = true;
                
                int duration = mp.getDuration();
                seekBar.setMax(duration);
                updateDuration(duration);
                
                playButton.setEnabled(true);
                Toast.makeText(AudioPlayerActivity.this, "Audio ready to play", 
                    Toast.LENGTH_SHORT).show();
            }
        });
        
        mediaPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                isPlaying = false;
                handler.removeCallbacks(updateSeekBar);
                
                playButton.setText("Play");
                playButton.setEnabled(true);
                pauseButton.setEnabled(false);
                
                // Reset to beginning
                seekBar.setProgress(0);
                updateCurrentTime(0);
                
                Toast.makeText(AudioPlayerActivity.this, "Playback completed", 
                    Toast.LENGTH_SHORT).show();
            }
        });
        
        mediaPlayer.setOnErrorListener(new MediaPlayer.OnErrorListener() {
            @Override
            public boolean onError(MediaPlayer mp, int what, int extra) {
                String errorMessage = getErrorMessage(what, extra);
                Toast.makeText(AudioPlayerActivity.this, errorMessage, Toast.LENGTH_LONG).show();
                
                resetPlayer();
                return true;
            }
        });
        
        // Load audio file
        loadAudioFile();
    }
    
    private void loadAudioFile() {
        try {
            // Load from assets
            AssetFileDescriptor afd = getAssets().openFd("sample_audio.mp3");
            mediaPlayer.setDataSource(afd.getFileDescriptor(), 
                afd.getStartOffset(), afd.getLength());
            afd.close();
            
            // Or load from URL
            // mediaPlayer.setDataSource("https://example.com/audio.mp3");
            
            // Or load from file
            // mediaPlayer.setDataSource("/path/to/audio/file.mp3");
            
            mediaPlayer.prepareAsync();
            
        } catch (IOException e) {
            Log.e("AudioPlayer", "Error loading audio file", e);
            Toast.makeText(this, "Error loading audio file", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void playAudio() {
        if (isPrepared && !isPlaying) {
            mediaPlayer.start();
            isPlaying = true;
            
            playButton.setText("Playing...");
            playButton.setEnabled(false);
            pauseButton.setEnabled(true);
            stopButton.setEnabled(true);
            
            handler.post(updateSeekBar);
        }
    }
    
    private void pauseAudio() {
        if (isPlaying) {
            mediaPlayer.pause();
            isPlaying = false;
            
            playButton.setText("Resume");
            playButton.setEnabled(true);
            pauseButton.setEnabled(false);
            
            handler.removeCallbacks(updateSeekBar);
        }
    }
    
    private void stopAudio() {
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            isPlaying = false;
            isPrepared = false;
            
            handler.removeCallbacks(updateSeekBar);
            
            playButton.setText("Play");
            playButton.setEnabled(false);
            pauseButton.setEnabled(false);
            stopButton.setEnabled(false);
            
            seekBar.setProgress(0);
            updateCurrentTime(0);
            
            // Prepare again for next playback
            try {
                mediaPlayer.prepare();
            } catch (IOException e) {
                Log.e("AudioPlayer", "Error preparing after stop", e);
            }
        }
    }
    
    private void setupSeekBarUpdater() {
        updateSeekBar = new Runnable() {
            @Override
            public void run() {
                if (mediaPlayer != null && isPlaying) {
                    int currentPosition = mediaPlayer.getCurrentPosition();
                    seekBar.setProgress(currentPosition);
                    updateCurrentTime(currentPosition);
                    
                    handler.postDelayed(this, 1000); // Update every second
                }
            }
        };
    }
    
    private void updateCurrentTime(int milliseconds) {
        String timeString = formatTime(milliseconds);
        currentTimeText.setText(timeString);
    }
    
    private void updateDuration(int milliseconds) {
        String timeString = formatTime(milliseconds);
        durationText.setText(timeString);
    }
    
    private String formatTime(int milliseconds) {
        int seconds = milliseconds / 1000;
        int minutes = seconds / 60;
        seconds = seconds % 60;
        
        return String.format(Locale.getDefault(), "%02d:%02d", minutes, seconds);
    }
    
    private String getErrorMessage(int what, int extra) {
        String errorMessage = "Unknown error occurred";
        
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
                errorMessage += " (Unsupported format)";
                break;
            case MediaPlayer.MEDIA_ERROR_TIMED_OUT:
                errorMessage += " (Connection timed out)";
                break;
        }
        
        return errorMessage;
    }
    
    private void resetPlayer() {
        isPlaying = false;
        isPrepared = false;
        
        playButton.setText("Play");
        playButton.setEnabled(false);
        pauseButton.setEnabled(false);
        stopButton.setEnabled(false);
        
        seekBar.setProgress(0);
        updateCurrentTime(0);
        
        handler.removeCallbacks(updateSeekBar);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        if (mediaPlayer != null) {
            handler.removeCallbacks(updateSeekBar);
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        
        if (isPlaying) {
            pauseAudio();
        }
    }
}
```

## Advanced Audio Features

### Playlist Management
```java
public class PlaylistManager {
    
    public static class Track {
        String title;
        String artist;
        String filePath;
        Uri uri;
        long duration;
        
        public Track(String title, String artist, String filePath) {
            this.title = title;
            this.artist = artist;
            this.filePath = filePath;
        }
        
        public Track(String title, String artist, Uri uri) {
            this.title = title;
            this.artist = artist;
            this.uri = uri;
        }
    }
    
    private List<Track> playlist;
    private int currentTrackIndex;
    private PlaylistListener listener;
    
    public interface PlaylistListener {
        void onTrackChanged(Track track, int position);
        void onPlaylistCompleted();
    }
    
    public PlaylistManager(PlaylistListener listener) {
        this.playlist = new ArrayList<>();
        this.currentTrackIndex = 0;
        this.listener = listener;
    }
    
    public void addTrack(Track track) {
        playlist.add(track);
    }
    
    public void removeTrack(int index) {
        if (index >= 0 && index < playlist.size()) {
            playlist.remove(index);
            
            if (currentTrackIndex >= index && currentTrackIndex > 0) {
                currentTrackIndex--;
            }
        }
    }
    
    public Track getCurrentTrack() {
        if (!playlist.isEmpty() && currentTrackIndex >= 0 && currentTrackIndex < playlist.size()) {
            return playlist.get(currentTrackIndex);
        }
        return null;
    }
    
    public Track getNextTrack() {
        if (hasNextTrack()) {
            currentTrackIndex++;
            Track track = getCurrentTrack();
            if (listener != null) {
                listener.onTrackChanged(track, currentTrackIndex);
            }
            return track;
        }
        return null;
    }
    
    public Track getPreviousTrack() {
        if (hasPreviousTrack()) {
            currentTrackIndex--;
            Track track = getCurrentTrack();
            if (listener != null) {
                listener.onTrackChanged(track, currentTrackIndex);
            }
            return track;
        }
        return null;
    }
    
    public boolean hasNextTrack() {
        return currentTrackIndex < playlist.size() - 1;
    }
    
    public boolean hasPreviousTrack() {
        return currentTrackIndex > 0;
    }
    
    public void setCurrentTrack(int index) {
        if (index >= 0 && index < playlist.size()) {
            currentTrackIndex = index;
            Track track = getCurrentTrack();
            if (listener != null) {
                listener.onTrackChanged(track, currentTrackIndex);
            }
        }
    }
    
    public List<Track> getPlaylist() {
        return new ArrayList<>(playlist);
    }
    
    public void shuffle() {
        Collections.shuffle(playlist);
        currentTrackIndex = 0;
        if (listener != null && !playlist.isEmpty()) {
            listener.onTrackChanged(getCurrentTrack(), currentTrackIndex);
        }
    }
    
    public void clear() {
        playlist.clear();
        currentTrackIndex = 0;
    }
    
    public int size() {
        return playlist.size();
    }
    
    public boolean isEmpty() {
        return playlist.isEmpty();
    }
}
```

### Audio Equalizer
```java
public class AudioEqualizerActivity extends AppCompatActivity {
    
    private MediaPlayer mediaPlayer;
    private Equalizer equalizer;
    private BassBoost bassBoost;
    private Virtualizer virtualizer;
    
    private SeekBar[] equalizerSeekBars;
    private SeekBar bassBoostSeekBar;
    private SeekBar virtualizerSeekBar;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_audio_equalizer);
        
        setupMediaPlayer();
        setupAudioEffects();
        setupUI();
    }
    
    private void setupMediaPlayer() {
        mediaPlayer = new MediaPlayer();
        
        try {
            // Load audio file
            AssetFileDescriptor afd = getAssets().openFd("sample_audio.mp3");
            mediaPlayer.setDataSource(afd.getFileDescriptor(), 
                afd.getStartOffset(), afd.getLength());
            afd.close();
            
            mediaPlayer.setOnPreparedListener(mp -> {
                setupAudioEffects();
                mp.start();
            });
            
            mediaPlayer.prepareAsync();
            
        } catch (IOException e) {
            Log.e("Equalizer", "Error loading audio", e);
        }
    }
    
    private void setupAudioEffects() {
        if (mediaPlayer.getAudioSessionId() != 0) {
            // Initialize equalizer
            equalizer = new Equalizer(0, mediaPlayer.getAudioSessionId());
            equalizer.setEnabled(true);
            
            // Initialize bass boost
            bassBoost = new BassBoost(0, mediaPlayer.getAudioSessionId());
            bassBoost.setEnabled(true);
            
            // Initialize virtualizer
            virtualizer = new Virtualizer(0, mediaPlayer.getAudioSessionId());
            virtualizer.setEnabled(true);
        }
    }
    
    private void setupUI() {
        LinearLayout equalizerLayout = findViewById(R.id.equalizerLayout);
        
        if (equalizer != null) {
            // Create equalizer controls
            short numberOfBands = equalizer.getNumberOfBands();
            equalizerSeekBars = new SeekBar[numberOfBands];
            
            for (short i = 0; i < numberOfBands; i++) {
                final short bandIndex = i;
                
                // Create frequency label
                TextView frequencyLabel = new TextView(this);
                int frequency = equalizer.getCenterFreq(bandIndex) / 1000;
                frequencyLabel.setText(frequency + " Hz");
                frequencyLabel.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
                equalizerLayout.addView(frequencyLabel);
                
                // Create seek bar for band
                SeekBar seekBar = new SeekBar(this);
                equalizerSeekBars[i] = seekBar;
                
                short[] bandLevelRange = equalizer.getBandLevelRange();
                seekBar.setMax(bandLevelRange[1] - bandLevelRange[0]);
                seekBar.setProgress((equalizer.getBandLevel(bandIndex) - bandLevelRange[0]));
                
                seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                    @Override
                    public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                        if (fromUser) {
                            short level = (short) (progress + bandLevelRange[0]);
                            equalizer.setBandLevel(bandIndex, level);
                        }
                    }
                    
                    @Override
                    public void onStartTrackingTouch(SeekBar seekBar) {}
                    
                    @Override
                    public void onStopTrackingTouch(SeekBar seekBar) {}
                });
                
                equalizerLayout.addView(seekBar);
            }
        }
        
        // Setup bass boost control
        bassBoostSeekBar = findViewById(R.id.bassBoostSeekBar);
        if (bassBoost != null) {
            bassBoostSeekBar.setMax(1000);
            bassBoostSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    if (fromUser) {
                        bassBoost.setStrength((short) progress);
                    }
                }
                
                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {}
                
                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {}
            });
        }
        
        // Setup virtualizer control
        virtualizerSeekBar = findViewById(R.id.virtualizerSeekBar);
        if (virtualizer != null) {
            virtualizerSeekBar.setMax(1000);
            virtualizerSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    if (fromUser) {
                        virtualizer.setStrength((short) progress);
                    }
                }
                
                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {}
                
                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {}
            });
        }
        
        // Preset buttons
        setupPresetButtons();
    }
    
    private void setupPresetButtons() {
        Button rockButton = findViewById(R.id.rockPresetButton);
        Button jazzButton = findViewById(R.id.jazzPresetButton);
        Button popButton = findViewById(R.id.popPresetButton);
        Button classicalButton = findViewById(R.id.classicalPresetButton);
        
        rockButton.setOnClickListener(v -> applyRockPreset());
        jazzButton.setOnClickListener(v -> applyJazzPreset());
        popButton.setOnClickListener(v -> applyPopPreset());
        classicalButton.setOnClickListener(v -> applyClassicalPreset());
    }
    
    private void applyRockPreset() {
        if (equalizer != null) {
            // Rock preset: boost bass and treble
            short[] bandLevelRange = equalizer.getBandLevelRange();
            short numberOfBands = equalizer.getNumberOfBands();
            
            for (short i = 0; i < numberOfBands; i++) {
                float frequency = equalizer.getCenterFreq(i) / 1000f; // Convert to kHz
                
                short level;
                if (frequency < 250) {
                    level = (short) (bandLevelRange[1] * 0.8); // Boost bass
                } else if (frequency > 4000) {
                    level = (short) (bandLevelRange[1] * 0.6); // Boost treble
                } else {
                    level = (short) (bandLevelRange[1] * 0.2); // Slight boost mids
                }
                
                equalizer.setBandLevel(i, level);
                equalizerSeekBars[i].setProgress(level - bandLevelRange[0]);
            }
            
            bassBoost.setStrength((short) 800);
            bassBoostSeekBar.setProgress(800);
        }
    }
    
    private void applyJazzPreset() {
        if (equalizer != null) {
            // Jazz preset: enhance mids, subtle bass and treble
            short[] bandLevelRange = equalizer.getBandLevelRange();
            short numberOfBands = equalizer.getNumberOfBands();
            
            for (short i = 0; i < numberOfBands; i++) {
                float frequency = equalizer.getCenterFreq(i) / 1000f;
                
                short level;
                if (frequency < 250) {
                    level = (short) (bandLevelRange[1] * 0.3); // Subtle bass
                } else if (frequency >= 250 && frequency <= 2000) {
                    level = (short) (bandLevelRange[1] * 0.7); // Enhance mids
                } else {
                    level = (short) (bandLevelRange[1] * 0.4); // Subtle treble
                }
                
                equalizer.setBandLevel(i, level);
                equalizerSeekBars[i].setProgress(level - bandLevelRange[0]);
            }
            
            virtualizer.setStrength((short) 600);
            virtualizerSeekBar.setProgress(600);
        }
    }
    
    private void applyPopPreset() {
        // Pop preset implementation
        // Balanced with slight bass and treble boost
    }
    
    private void applyClassicalPreset() {
        // Classical preset implementation
        // Natural sound with minimal processing
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        if (mediaPlayer != null) {
            mediaPlayer.release();
        }
        
        if (equalizer != null) {
            equalizer.release();
        }
        
        if (bassBoost != null) {
            bassBoost.release();
        }
        
        if (virtualizer != null) {
            virtualizer.release();
        }
    }
}
```

## Video Playback Implementation

### Custom Video Player
```java
public class CustomVideoPlayerActivity extends AppCompatActivity {
    
    private VideoView videoView;
    private ProgressBar progressBar;
    private View controlsOverlay;
    private ImageButton playPauseButton;
    private ImageButton fullscreenButton;
    private SeekBar videoSeekBar;
    private TextView currentTimeText, durationText;
    
    private Handler handler;
    private Runnable updateProgress;
    private boolean isControlsVisible = true;
    private boolean isFullscreen = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_custom_video_player);
        
        initializeViews();
        setupVideoPlayer();
        setupControls();
    }
    
    private void initializeViews() {
        videoView = findViewById(R.id.videoView);
        progressBar = findViewById(R.id.progressBar);
        controlsOverlay = findViewById(R.id.controlsOverlay);
        playPauseButton = findViewById(R.id.playPauseButton);
        fullscreenButton = findViewById(R.id.fullscreenButton);
        videoSeekBar = findViewById(R.id.videoSeekBar);
        currentTimeText = findViewById(R.id.currentTimeText);
        durationText = findViewById(R.id.durationText);
        
        handler = new Handler(Looper.getMainLooper());
    }
    
    private void setupVideoPlayer() {
        String videoPath = getIntent().getStringExtra("video_path");
        if (videoPath != null) {
            Uri videoUri = Uri.parse(videoPath);
            videoView.setVideoURI(videoUri);
        }
        
        videoView.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                progressBar.setVisibility(View.GONE);
                controlsOverlay.setVisibility(View.VISIBLE);
                
                int duration = mp.getDuration();
                videoSeekBar.setMax(duration);
                updateDurationText(duration);
                
                // Auto-start playback
                videoView.start();
                playPauseButton.setImageResource(R.drawable.ic_pause);
                
                startProgressUpdate();
            }
        });
        
        videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                playPauseButton.setImageResource(R.drawable.ic_play);
                stopProgressUpdate();
                
                // Reset to beginning
                videoView.seekTo(0);
                videoSeekBar.setProgress(0);
                updateCurrentTimeText(0);
            }
        });
        
        videoView.setOnErrorListener(new MediaPlayer.OnErrorListener() {
            @Override
            public boolean onError(MediaPlayer mp, int what, int extra) {
                progressBar.setVisibility(View.GONE);
                Toast.makeText(CustomVideoPlayerActivity.this, 
                    "Error playing video", Toast.LENGTH_SHORT).show();
                return true;
            }
        });
        
        // Touch listener for showing/hiding controls
        videoView.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                toggleControlsVisibility();
            }
            return true;
        });
    }
    
    private void setupControls() {
        playPauseButton.setOnClickListener(v -> togglePlayPause());
        fullscreenButton.setOnClickListener(v -> toggleFullscreen());
        
        videoSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    videoView.seekTo(progress);
                    updateCurrentTimeText(progress);
                }
            }
            
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                stopProgressUpdate();
            }
            
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                if (videoView.isPlaying()) {
                    startProgressUpdate();
                }
            }
        });
        
        // Auto-hide controls after 3 seconds
        setupAutoHideControls();
    }
    
    private void togglePlayPause() {
        if (videoView.isPlaying()) {
            videoView.pause();
            playPauseButton.setImageResource(R.drawable.ic_play);
            stopProgressUpdate();
        } else {
            videoView.start();
            playPauseButton.setImageResource(R.drawable.ic_pause);
            startProgressUpdate();
        }
        
        resetAutoHideTimer();
    }
    
    private void toggleFullscreen() {
        if (isFullscreen) {
            exitFullscreen();
        } else {
            enterFullscreen();
        }
    }
    
    private void enterFullscreen() {
        isFullscreen = true;
        
        // Hide system UI
        getWindow().getDecorView().setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_IMMERSIVE
            | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
            | View.SYSTEM_UI_FLAG_FULLSCREEN);
        
        // Change video view layout
        ViewGroup.LayoutParams params = videoView.getLayoutParams();
        params.width = ViewGroup.LayoutParams.MATCH_PARENT;
        params.height = ViewGroup.LayoutParams.MATCH_PARENT;
        videoView.setLayoutParams(params);
        
        fullscreenButton.setImageResource(R.drawable.ic_fullscreen_exit);
    }
    
    private void exitFullscreen() {
        isFullscreen = false;
        
        // Show system UI
        getWindow().getDecorView().setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
        
        // Restore video view layout
        ViewGroup.LayoutParams params = videoView.getLayoutParams();
        params.width = ViewGroup.LayoutParams.MATCH_PARENT;
        params.height = 0; // Use layout weight
        videoView.setLayoutParams(params);
        
        fullscreenButton.setImageResource(R.drawable.ic_fullscreen);
    }
    
    private void toggleControlsVisibility() {
        if (isControlsVisible) {
            hideControls();
        } else {
            showControls();
        }
    }
    
    private void showControls() {
        controlsOverlay.setVisibility(View.VISIBLE);
        isControlsVisible = true;
        resetAutoHideTimer();
    }
    
    private void hideControls() {
        if (!isFullscreen) return; // Don't hide in windowed mode
        
        controlsOverlay.setVisibility(View.GONE);
        isControlsVisible = false;
        handler.removeCallbacks(hideControlsRunnable);
    }
    
    private Runnable hideControlsRunnable = new Runnable() {
        @Override
        public void run() {
            if (videoView.isPlaying() && isFullscreen) {
                hideControls();
            }
        }
    };
    
    private void setupAutoHideControls() {
        resetAutoHideTimer();
    }
    
    private void resetAutoHideTimer() {
        handler.removeCallbacks(hideControlsRunnable);
        handler.postDelayed(hideControlsRunnable, 3000); // Hide after 3 seconds
    }
    
    private void startProgressUpdate() {
        updateProgress = new Runnable() {
            @Override
            public void run() {
                if (videoView.isPlaying()) {
                    int currentPosition = videoView.getCurrentPosition();
                    videoSeekBar.setProgress(currentPosition);
                    updateCurrentTimeText(currentPosition);
                    
                    handler.postDelayed(this, 1000);
                }
            }
        };
        
        handler.post(updateProgress);
    }
    
    private void stopProgressUpdate() {
        if (updateProgress != null) {
            handler.removeCallbacks(updateProgress);
        }
    }
    
    private void updateCurrentTimeText(int milliseconds) {
        String timeString = formatTime(milliseconds);
        currentTimeText.setText(timeString);
    }
    
    private void updateDurationText(int milliseconds) {
        String timeString = formatTime(milliseconds);
        durationText.setText(timeString);
    }
    
    private String formatTime(int milliseconds) {
        int totalSeconds = milliseconds / 1000;
        int hours = totalSeconds / 3600;
        int minutes = (totalSeconds % 3600) / 60;
        int seconds = totalSeconds % 60;
        
        if (hours > 0) {
            return String.format(Locale.getDefault(), "%d:%02d:%02d", hours, minutes, seconds);
        } else {
            return String.format(Locale.getDefault(), "%02d:%02d", minutes, seconds);
        }
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        
        if (videoView.isPlaying()) {
            videoView.pause();
            playPauseButton.setImageResource(R.drawable.ic_play);
            stopProgressUpdate();
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        stopProgressUpdate();
        handler.removeCallbacks(hideControlsRunnable);
        
        if (videoView != null) {
            videoView.stopPlayback();
        }
    }
    
    @Override
    public void onBackPressed() {
        if (isFullscreen) {
            exitFullscreen();
        } else {
            super.onBackPressed();
        }
    }
}
```

## Audio Recording

### Simple Audio Recorder
```java
public class AudioRecorderActivity extends AppCompatActivity {
    
    private MediaRecorder mediaRecorder;
    private String audioFilePath;
    private Button recordButton, stopButton, playButton;
    private TextView statusText;
    
    private boolean isRecording = false;
    private MediaPlayer playbackPlayer;
    
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_audio_recorder);
        
        initializeViews();
        checkPermissions();
    }
    
    private void initializeViews() {
        recordButton = findViewById(R.id.recordButton);
        stopButton = findViewById(R.id.stopButton);
        playButton = findViewById(R.id.playButton);
        statusText = findViewById(R.id.statusText);
        
        recordButton.setOnClickListener(v -> startRecording());
        stopButton.setOnClickListener(v -> stopRecording());
        playButton.setOnClickListener(v -> playRecording());
        
        stopButton.setEnabled(false);
        playButton.setEnabled(false);
    }
    
    private void checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.RECORD_AUDIO}, 
                REQUEST_RECORD_AUDIO_PERMISSION);
        }
    }
    
    private void startRecording() {
        if (isRecording) return;
        
        try {
            // Create output file
            audioFilePath = createAudioFile();
            
            // Setup MediaRecorder
            mediaRecorder = new MediaRecorder();
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.AAC_ADTS);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setAudioSamplingRate(44100);
            mediaRecorder.setAudioEncodingBitRate(128000);
            mediaRecorder.setOutputFile(audioFilePath);
            
            mediaRecorder.prepare();
            mediaRecorder.start();
            
            isRecording = true;
            
            recordButton.setEnabled(false);
            stopButton.setEnabled(true);
            playButton.setEnabled(false);
            
            statusText.setText("Recording...");
            
        } catch (IOException e) {
            Log.e("AudioRecorder", "Error starting recording", e);
            statusText.setText("Error starting recording");
        }
    }
    
    private void stopRecording() {
        if (!isRecording) return;
        
        try {
            mediaRecorder.stop();
            mediaRecorder.release();
            mediaRecorder = null;
            
            isRecording = false;
            
            recordButton.setEnabled(true);
            stopButton.setEnabled(false);
            playButton.setEnabled(true);
            
            statusText.setText("Recording saved: " + audioFilePath);
            
        } catch (RuntimeException e) {
            Log.e("AudioRecorder", "Error stopping recording", e);
            statusText.setText("Error stopping recording");
        }
    }
    
    private void playRecording() {
        if (audioFilePath == null) return;
        
        try {
            if (playbackPlayer != null) {
                playbackPlayer.release();
            }
            
            playbackPlayer = new MediaPlayer();
            playbackPlayer.setDataSource(audioFilePath);
            
            playbackPlayer.setOnPreparedListener(mp -> {
                mp.start();
                statusText.setText("Playing recording...");
                playButton.setText("Playing...");
                playButton.setEnabled(false);
            });
            
            playbackPlayer.setOnCompletionListener(mp -> {
                statusText.setText("Playback completed");
                playButton.setText("Play");
                playButton.setEnabled(true);
            });
            
            playbackPlayer.setOnErrorListener((mp, what, extra) -> {
                statusText.setText("Error playing recording");
                playButton.setText("Play");
                playButton.setEnabled(true);
                return true;
            });
            
            playbackPlayer.prepareAsync();
            
        } catch (IOException e) {
            Log.e("AudioRecorder", "Error playing recording", e);
            statusText.setText("Error playing recording");
        }
    }
    
    private String createAudioFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            .format(new Date());
        String audioFileName = "AUDIO_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_MUSIC);
        
        File audioFile = File.createTempFile(audioFileName, ".aac", storageDir);
        return audioFile.getAbsolutePath();
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        if (mediaRecorder != null) {
            mediaRecorder.release();
            mediaRecorder = null;
        }
        
        if (playbackPlayer != null) {
            playbackPlayer.release();
            playbackPlayer = null;
        }
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, 
                                         int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Audio recording permission granted", 
                    Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Audio recording permission denied", 
                    Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }
}
```

## Background Audio Playback

### Audio Service for Background Playback
```java
public class AudioService extends Service {
    
    private static final String CHANNEL_ID = "AudioServiceChannel";
    private static final int NOTIFICATION_ID = 1;
    
    private MediaPlayer mediaPlayer;
    private PlaylistManager playlistManager;
    private MediaSession mediaSession;
    private PlaybackStateCompat.Builder playbackStateBuilder;
    
    public static final String ACTION_PLAY = "com.example.ACTION_PLAY";
    public static final String ACTION_PAUSE = "com.example.ACTION_PAUSE";
    public static final String ACTION_STOP = "com.example.ACTION_STOP";
    public static final String ACTION_NEXT = "com.example.ACTION_NEXT";
    public static final String ACTION_PREVIOUS = "com.example.ACTION_PREVIOUS";
    
    @Override
    public void onCreate() {
        super.onCreate();
        
        createNotificationChannel();
        setupMediaSession();
        setupPlaylistManager();
    }
    
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            String action = intent.getAction();
            
            switch (action) {
                case ACTION_PLAY:
                    playAudio();
                    break;
                case ACTION_PAUSE:
                    pauseAudio();
                    break;
                case ACTION_STOP:
                    stopAudio();
                    break;
                case ACTION_NEXT:
                    playNext();
                    break;
                case ACTION_PREVIOUS:
                    playPrevious();
                    break;
            }
        }
        
        return START_STICKY; // Restart service if killed
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        return new AudioServiceBinder();
    }
    
    public class AudioServiceBinder extends Binder {
        public AudioService getService() {
            return AudioService.this;
        }
    }
    
    private void setupMediaSession() {
        mediaSession = new MediaSession(this, "AudioService");
        playbackStateBuilder = new PlaybackStateCompat.Builder();
        
        mediaSession.setCallback(new MediaSession.Callback() {
            @Override
            public void onPlay() {
                playAudio();
            }
            
            @Override
            public void onPause() {
                pauseAudio();
            }
            
            @Override
            public void onStop() {
                stopAudio();
            }
            
            @Override
            public void onSkipToNext() {
                playNext();
            }
            
            @Override
            public void onSkipToPrevious() {
                playPrevious();
            }
        });
        
        mediaSession.setActive(true);
    }
    
    private void setupPlaylistManager() {
        playlistManager = new PlaylistManager(new PlaylistManager.PlaylistListener() {
            @Override
            public void onTrackChanged(PlaylistManager.Track track, int position) {
                loadAndPlayTrack(track);
                updateNotification();
            }
            
            @Override
            public void onPlaylistCompleted() {
                stopAudio();
            }
        });
    }
    
    private void playAudio() {
        if (mediaPlayer != null && !mediaPlayer.isPlaying()) {
            mediaPlayer.start();
            updatePlaybackState(PlaybackStateCompat.STATE_PLAYING);
            updateNotification();
        }
    }
    
    private void pauseAudio() {
        if (mediaPlayer != null && mediaPlayer.isPlaying()) {
            mediaPlayer.pause();
            updatePlaybackState(PlaybackStateCompat.STATE_PAUSED);
            updateNotification();
        }
    }
    
    private void stopAudio() {
        if (mediaPlayer != null) {
            mediaPlayer.stop();
            mediaPlayer.release();
            mediaPlayer = null;
        }
        
        updatePlaybackState(PlaybackStateCompat.STATE_STOPPED);
        stopForeground(true);
        stopSelf();
    }
    
    private void playNext() {
        PlaylistManager.Track nextTrack = playlistManager.getNextTrack();
        if (nextTrack != null) {
            loadAndPlayTrack(nextTrack);
        }
    }
    
    private void playPrevious() {
        PlaylistManager.Track previousTrack = playlistManager.getPreviousTrack();
        if (previousTrack != null) {
            loadAndPlayTrack(previousTrack);
        }
    }
    
    private void loadAndPlayTrack(PlaylistManager.Track track) {
        try {
            if (mediaPlayer != null) {
                mediaPlayer.release();
            }
            
            mediaPlayer = new MediaPlayer();
            
            if (track.uri != null) {
                mediaPlayer.setDataSource(this, track.uri);
            } else {
                mediaPlayer.setDataSource(track.filePath);
            }
            
            mediaPlayer.setOnPreparedListener(mp -> {
                mp.start();
                updatePlaybackState(PlaybackStateCompat.STATE_PLAYING);
                updateNotification();
            });
            
            mediaPlayer.setOnCompletionListener(mp -> {
                if (playlistManager.hasNextTrack()) {
                    playNext();
                } else {
                    stopAudio();
                }
            });
            
            mediaPlayer.prepareAsync();
            
        } catch (IOException e) {
            Log.e("AudioService", "Error loading track", e);
        }
    }
    
    private void updatePlaybackState(int state) {
        PlaybackStateCompat playbackState = playbackStateBuilder
            .setState(state, mediaPlayer != null ? mediaPlayer.getCurrentPosition() : 0, 1.0f)
            .setActions(PlaybackStateCompat.ACTION_PLAY | PlaybackStateCompat.ACTION_PAUSE |
                       PlaybackStateCompat.ACTION_SKIP_TO_NEXT | PlaybackStateCompat.ACTION_SKIP_TO_PREVIOUS)
            .build();
        
        mediaSession.setPlaybackState(playbackState);
    }
    
    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Audio Playback",
                NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Controls for audio playback");
            
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(channel);
        }
    }
    
    private void updateNotification() {
        PlaylistManager.Track currentTrack = playlistManager.getCurrentTrack();
        if (currentTrack == null) return;
        
        boolean isPlaying = mediaPlayer != null && mediaPlayer.isPlaying();
        
        Intent playPauseIntent = new Intent(this, AudioService.class);
        playPauseIntent.setAction(isPlaying ? ACTION_PAUSE : ACTION_PLAY);
        PendingIntent playPausePendingIntent = PendingIntent.getService(
            this, 0, playPauseIntent, PendingIntent.FLAG_UPDATE_CURRENT);
        
        Intent nextIntent = new Intent(this, AudioService.class);
        nextIntent.setAction(ACTION_NEXT);
        PendingIntent nextPendingIntent = PendingIntent.getService(
            this, 0, nextIntent, PendingIntent.FLAG_UPDATE_CURRENT);
        
        Intent previousIntent = new Intent(this, AudioService.class);
        previousIntent.setAction(ACTION_PREVIOUS);
        PendingIntent previousPendingIntent = PendingIntent.getService(
            this, 0, previousIntent, PendingIntent.FLAG_UPDATE_CURRENT);
        
        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(currentTrack.title)
            .setContentText(currentTrack.artist)
            .setSmallIcon(R.drawable.ic_music_note)
            .addAction(R.drawable.ic_skip_previous, "Previous", previousPendingIntent)
            .addAction(isPlaying ? R.drawable.ic_pause : R.drawable.ic_play, 
                      isPlaying ? "Pause" : "Play", playPausePendingIntent)
            .addAction(R.drawable.ic_skip_next, "Next", nextPendingIntent)
            .setStyle(new androidx.media.app.NotificationCompat.MediaStyle()
                     .setMediaSession(mediaSession.getSessionToken())
                     .setShowActionsInCompactView(0, 1, 2))
            .build();
        
        startForeground(NOTIFICATION_ID, notification);
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
        
        if (mediaSession != null) {
            mediaSession.release();
        }
    }
}
```

## Best Practices

### Media Playback Best Practices
- **Handle audio focus properly**: Request and abandon audio focus appropriately
- **Implement proper lifecycle management**: Pause/resume playback based on app state
- **Use background services**: For continuous audio playback
- **Optimize memory usage**: Release MediaPlayer resources when not needed
- **Handle network conditions**: Implement buffering strategies for streaming

### Performance Optimization
- **Use appropriate codecs**: Choose efficient audio/video formats
- **Implement caching**: Cache frequently played media files
- **Handle large files efficiently**: Stream large videos instead of loading entirely
- **Optimize UI updates**: Update progress bars efficiently without blocking UI

### User Experience
- **Provide visual feedback**: Show loading states and progress indicators
- **Implement media controls**: Standard play/pause/seek functionality
- **Handle interruptions gracefully**: Phone calls, notifications, etc.
- **Support media sessions**: Integrate with system media controls
- **Persist playback state**: Remember user's position in media

Understanding audio and video playback in Android enables creating rich multimedia applications with professional-quality media experiences. Proper implementation of MediaPlayer, audio focus, and background services ensures smooth and user-friendly media playback.

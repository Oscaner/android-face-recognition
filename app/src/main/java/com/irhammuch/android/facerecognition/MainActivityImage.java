package com.irhammuch.android.facerecognition;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;

import androidx.appcompat.app.AppCompatActivity;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

public class MainActivityImage extends AppCompatActivity {

    private static final Logger logger = Logger.getLogger(MainActivity.class.getName());

    private static final String TAG = "MainActivityImage";

    // UI
    private Button mFabActionBt;

    private Interpreter tfLite;

    private FaceDetector faceDetector;

    private final HashMap<String, SimilarityClassifier.Recognition> registered = new HashMap<>(); //saved Faces

    private final HashMap<String, InputImage> inputImages = new HashMap<>();

    private final HashMap<String, List<String>> imageTags = new HashMap<>(); // image tags.

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_image);

        loadModel();

        faceDetector = FaceDetection.getClient(new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .build()
        );

        setupUI();
    }

    protected void setupUI() {
        mFabActionBt = (Button) findViewById(R.id.start);
        mFabActionBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(MainActivityImage.this, "Start", Toast.LENGTH_SHORT).show();
                try {
                    for (String f: getAssets().list("face_images/")) {
                        logger.info("Processing face_images/" + f);
                        analyze("face_images/" + f);
                    }
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private Boolean analyze(@NonNull String image_path) throws InterruptedException {
        InputImage image = null;
        try {
            image = inputImages.get(image_path);
            if (image == null) {
                image = InputImage.fromBitmap(BitmapFactory.decodeStream(getAssets().open(image_path)), 0);
                inputImages.put(image_path, image);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert image != null;

        return faceDetector.process(image)
                .addOnSuccessListener(faces -> onSuccessListener(faces, image_path))
                .addOnFailureListener(e -> e.printStackTrace())
                .addOnCompleteListener(task -> logger.info("Done of " + image_path)).isComplete();
    }

    private void onSuccessListener(List<Face> faces, String image_path) {
        logger.info("Find " + faces.size() + " faces.");

        for (Face face: faces) {
            try {
                Rect boundingBox = face.getBoundingBox();

                Bitmap bitmap = ImageUtils.croppedFace(
                        BitmapFactory.decodeStream(getAssets().open(image_path)),
                        inputImages.get(image_path).getRotationDegrees(),
                        boundingBox
                );

                String name = ImageUtils.recognizeImage(bitmap, tfLite, registered);

                List<String> tags = imageTags.getOrDefault(image_path, new ArrayList<>());
                tags.add(name);
                imageTags.put(image_path, tags);

                System.out.println("Image Tags: " + imageTags.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /** Model loader */
    @SuppressWarnings("deprecation")
    private void loadModel() {
        try {
            //model name
            String modelFile = "mobile_face_net.tflite";
            tfLite = new Interpreter(loadModelFile(MainActivityImage.this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
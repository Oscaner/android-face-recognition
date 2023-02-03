package com.irhammuch.android.facerecognition;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import androidx.annotation.NonNull;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.dexafree.materialList.card.Card;
import com.dexafree.materialList.card.CardProvider;
import com.dexafree.materialList.view.MaterialListView;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.squareup.picasso.RequestCreator;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class MainActivityImage extends AppCompatActivity {

    private static final Logger logger = Logger.getLogger(MainActivity.class.getName());

    private static final String TAG = "MainActivityImage";

    // UI
    private RelativeLayout mRelativeLayout;
    private LinearLayout mLinearLayout;
    private HashMap<String, MaterialListView> materialListViews = new HashMap<>();
    private Button mStartBtn;

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

        mStartBtn.setVisibility(View.GONE);
        start();
    }

    protected void setupUI() {
        mRelativeLayout = (RelativeLayout) findViewById(R.id.relative_layout);
        mLinearLayout = (LinearLayout) findViewById(R.id.linear_layout);
        mStartBtn = (Button) findViewById(R.id.start);

        materialListViews.put("empty", createMaterialListView());
        mLinearLayout.addView(materialListViews.get("empty"));

        mStartBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                start();
            }
        });
    }

    private void start() {
        final String[] columns = {MediaStore.Images.Media.DATA, MediaStore.Images.Media._ID};
        final String orderBy = MediaStore.Images.Media._ID;

        // Stores all the images from the gallery in Cursor.
        Cursor cursor = getContentResolver().query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                columns,
                null,
                null,
                orderBy
        );

        // Total number of images.
        int count = cursor.getCount();

        System.out.println("Count: " + count);

        // Create an array to store path to all the images.
        ArrayList<String> arrPath = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            cursor.moveToPosition(i);
            int dataColumnIndex = cursor.getColumnIndex(MediaStore.Images.Media.DATA);
            // Store the path of the image.
            arrPath.add(0, cursor.getString(dataColumnIndex));
        }

        for (String each: arrPath) {
            logger.info("Path: " + each);

            try {
                analyze(each);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private Boolean analyze(@NonNull String image_path) throws InterruptedException {
        InputImage image = null;
        try {
            image = inputImages.get(image_path);
            if (image == null) {
                image = InputImage.fromFilePath(getApplicationContext(), Uri.fromFile(new File(image_path)));
                inputImages.put(image_path, image);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        assert image != null;

        return faceDetector.process(image)
                .addOnSuccessListener(faces -> onSuccessListener(faces, image_path))
                .addOnFailureListener(e -> e.printStackTrace())
                .addOnCompleteListener(task -> updateListView()).isComplete();
    }

    private void onSuccessListener(List<Face> faces, String image_path) {
        logger.info("Find " + faces.size() + " faces.");

        if (faces.size() == 0) {
            imageTags.put(image_path, null);
        }

        for (Face face: faces) {
            Rect boundingBox = face.getBoundingBox();

            Bitmap bitmap = ImageUtils.croppedFace(
                    BitmapFactory.decodeFile(image_path),
                    inputImages.get(image_path).getRotationDegrees(),
                    boundingBox
            );

            String name = ImageUtils.recognizeImage(bitmap, tfLite, registered);

            List<String> tags = null;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                tags = imageTags.getOrDefault(image_path, new ArrayList<>());
            }
            tags.add(name);
            imageTags.put(image_path, tags);
        }

        logger.info("Done of " + image_path);
    }

    private void updateListView() {
        for (MaterialListView each: materialListViews.values()) {
            each.getAdapter().clearAll();
        }

        for (String tag: registered.keySet()) {
            if (materialListViews.get(tag) == null) {
                materialListViews.put(tag, createMaterialListView());
                mLinearLayout.addView(materialListViews.get(tag));
            }
        }

        for (Map.Entry<String, List<String>> entry: imageTags.entrySet()) {
            if (entry.getValue() == null) {
                materialListViews.get("empty").getAdapter().add(createCard("empty", entry));
            } else {
                for (String tag: entry.getValue()) {
                    logger.info(tag);
                    materialListViews.get(tag).getAdapter().add(createCard(tag, entry));
                }
            }
        }
    }

    private MaterialListView createMaterialListView() {
        MaterialListView view = new MaterialListView(this);

        view.setLayoutParams(
                new ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                )
        );

        view.setLayoutManager(
                new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        );

        GradientDrawable border = new GradientDrawable();
        border.setColor(0xFFFFFFFF);
        border.setStroke(1, 0xFF000000);

        view.setBackground(border);
        return view;
    }

    private Card createCard(String tag, Map.Entry<String, List<String>> entry) {
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = 6;
        Bitmap bitmap = BitmapFactory.decodeFile(
                entry.getKey(), opts
        );

        return new Card.Builder(this)
                .withProvider(new CardProvider())
                .setLayout(R.layout.material_basic_image_buttons_card_layout)
                .setTitle(entry.getKey().substring(entry.getKey().lastIndexOf("/") + 1))
                .setTitleGravity(Gravity.END)
                .setDescription(String.valueOf(entry.getValue() == null ? "No faces" : "Find \"" + tag + "\" in " +  entry.getValue()))
                .setDescriptionGravity(Gravity.END)
                .setDrawable(new BitmapDrawable(getResources(), bitmap))
                .setDrawableConfiguration(new CardProvider.OnImageConfigListener() {
                    @Override
                    public void onImageConfigure(@NonNull RequestCreator requestCreator) {
                        requestCreator.fit();
                    }
                })
                .endConfig()
                .build();
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

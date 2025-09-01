from cnn_jax import MLP
import time
import joblib


if __name__ == "__main__":
    mlp = MLP()

    print("NONE")
    print("----------")
    start = time.perf_counter()
    mlp.flow()
    end = time.perf_counter()
    elapsed = end - start
    print(
        f"Final Accuracy at Epoch {mlp.epochs}, train_acc = {mlp.accuracy(mlp.train_images, mlp.train_labels)},"
        f"test_acc = {mlp.accuracy(mlp.test_images, mlp.test_labels)}"
    )
    joblib.dump(mlp, "jax_cnn_model.joblib")
    print("model saved")
    print(f"Time taken without jit: {elapsed:.6f} seconds")

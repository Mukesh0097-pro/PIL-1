"""
Test suite for PIL-VAE Engine
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient


def test_config():
    """Test configuration loading"""
    from app.core.config import settings

    assert settings.PROJECT_NAME == "indxai OS"
    assert settings.EMBEDDING_DIM == 384
    assert settings.LATENT_DIM == 24
    print("✅ Config test passed")


def test_pil_vae():
    """Test PIL-VAE functionality"""
    from app.core.pil_vae import PILVAE

    # Initialize
    vae = PILVAE(input_dim=384, latent_dim=24, hidden_dim=128)
    assert not vae.is_fitted, "VAE should not be fitted initially"

    # Fit
    X = np.random.randn(20, 384).astype(np.float32)
    vae.fit(X)
    assert vae.is_fitted, "VAE should be fitted after calling fit()"

    # Encode/Decode
    x = np.random.randn(384).astype(np.float32)
    z = vae.encode(x)
    assert z.shape == (24,), f"Expected latent shape (24,), got {z.shape}"

    x_rec = vae.decode(z)
    assert x_rec.shape == (384,), (
        f"Expected reconstruction shape (384,), got {x_rec.shape}"
    )

    # Generate
    samples = vae.generate(n_samples=5)
    assert samples.shape == (5, 384), (
        f"Expected samples shape (5, 384), got {samples.shape}"
    )

    # Reconstruction error
    error = vae.get_reconstruction_error(X[:5])
    assert error < float("inf"), "Reconstruction error should be finite"
    assert error >= 0, "Reconstruction error should be non-negative"

    print("✅ PIL-VAE test passed")


def test_memory_layer():
    """Test Memory Layer with FAISS"""
    from app.core.memory import MemoryLayer
    import os
    import time

    # Use test database
    test_db = "test_memory.db"
    memory = None

    try:
        memory = MemoryLayer(db_path=test_db, embedding_dim=384)

        # Add entries
        for i in range(5):
            vec = np.random.randn(384).astype(np.float32)
            memory.add(f"Test text {i}", vec, "test")

        # Check stats
        stats = memory.get_stats()
        assert stats["knowledge_entries"] == 5, (
            f"Expected 5 entries, got {stats['knowledge_entries']}"
        )
        assert stats["faiss_vectors"] == 5, (
            f"Expected 5 FAISS vectors, got {stats['faiss_vectors']}"
        )

        # Retrieve
        query = np.random.randn(384).astype(np.float32)
        results = memory.retrieve(query, top_k=3)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert all("text" in r and "score" in r for r in results), (
            "Results missing text or score"
        )

        # Clear
        memory.clear_knowledge()
        stats = memory.get_stats()
        assert stats["knowledge_entries"] == 0, "Knowledge should be cleared"

        print("✅ Memory Layer test passed")
    finally:
        # Close connection before cleanup
        if memory:
            memory.close()

        # Wait a bit for Windows to release the file
        time.sleep(0.5)

        # Cleanup
        try:
            if os.path.exists(test_db):
                os.remove(test_db)
            # Also remove WAL files if they exist
            for ext in ["-wal", "-shm"]:
                wal_file = test_db + ext
                if os.path.exists(wal_file):
                    os.remove(wal_file)
        except Exception:
            pass  # Ignore cleanup errors on Windows


def test_api_endpoints():
    """Test FastAPI endpoints"""
    from app.main import app

    client = TestClient(app)

    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["engine"] == "PIL-VAE Hybrid"

    # Test v1 health endpoint
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "memory" in data

    # Test stats endpoint
    response = client.get("/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "memory" in data
    assert "mode" in data

    # Test home page
    response = client.get("/")
    assert response.status_code == 200

    print("✅ API endpoints test passed")


def test_chat_endpoint():
    """Test chat functionality"""
    from app.main import app

    client = TestClient(app)

    # Test chat with a simple query
    response = client.post(
        "/chat", data={"query": "What is Python?", "mode": "assistant"}
    )
    assert response.status_code == 200

    # Check streaming response
    content = response.text
    assert len(content) > 0

    print("✅ Chat endpoint test passed")


def test_training_endpoint():
    """Test training functionality"""
    from app.main import app

    client = TestClient(app)

    # Test training
    response = client.post(
        "/train-knowledge",
        data={"text_data": "This is a test sentence. Another test sentence here."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Training started"

    print("✅ Training endpoint test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("Running PIL-VAE Engine Tests")
    print("=" * 50 + "\n")

    tests = [
        ("Config", test_config),
        ("PIL-VAE", test_pil_vae),
        ("Memory Layer", test_memory_layer),
        ("API Endpoints", test_api_endpoints),
        ("Chat Endpoint", test_chat_endpoint),
        ("Training Endpoint", test_training_endpoint),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n🔄 Testing {name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)

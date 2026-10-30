import pytest
from fastapi.testclient import TestClient

from alphaml_api.main import app
from .conftest import FakeModel

VALID = {
    "symbols": ["C", "H", "H", "H", "H"],
    "positions": [
        [0.0, 0.0, 0.0],
        [1.09, 0.0, 0.0],
        [-0.36, 1.03, 0.0],
        [-0.36, -0.51, 0.89],
        [-0.36, -0.51, -0.89],
    ],
    "model": "ccsd",
}


@pytest.fixture(scope="module")
def client():
    import alphaml_api.main as main

    original = main.load_model_data
    main.load_model_data = lambda: FakeModel()
    try:
        with TestClient(main.app) as test_client:
            yield test_client
    finally:
        main.load_model_data = original


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_shape(client):
    response = client.post("/api/predict", json=VALID)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["atomic_polarizabilities"]) == len(VALID["symbols"])
    assert all(len(row) == 9 for row in body["atomic_polarizabilities"])
    assert len(body["molecular_polarizability"]) == 9
    assert body["molecular_polarizability"] == [
        len(VALID["symbols"]),
        0.0,
        0.0,
        0.0,
        len(VALID["symbols"]),
        0.0,
        0.0,
        0.0,
        len(VALID["symbols"]),
    ]
    assert isinstance(body["compute_time"], float)


def test_too_many_atoms(client):
    payload = {
        "symbols": ["H"] * 2001,
        "positions": [[0.0, 0.0, 0.0]] * 2001,
        "model": "ccsd",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Sorry, this online app is limited to 2000 atoms in the input cell, "
        "while your structure has 2001 atoms."
    )


def test_disallowed_element(client):
    response = client.post(
        "/api/predict",
        json={"symbols": ["F"], "positions": [[0.0, 0.0, 0.0]], "model": "ccsd"},
    )
    assert response.status_code == 422
    assert "H, C, N, O, S, Cl" in response.json()["detail"]


def test_overlapping_atoms(client):
    response = client.post(
        "/api/predict",
        json={
            "symbols": ["H", "H"],
            "positions": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "model": "b3lyp",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "There are some overlapping atoms. Please check your input file."
    )


def test_symbols_positions_mismatch(client):
    response = client.post(
        "/api/predict",
        json={"symbols": ["H"], "positions": [], "model": "ccsd"},
    )
    assert response.status_code == 422


def test_unknown_model(client):
    response = client.post(
        "/api/predict",
        json={"symbols": ["H"], "positions": [[0.0, 0.0, 0.0]], "model": "nope"},
    )
    assert response.status_code == 422

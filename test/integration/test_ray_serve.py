import requests

from test.schemas import NameAgeOccupation


def test_name_query_ray_server_call(
    ontical_model_server_url: str, bob_thread_ids: list[str]
):
    """Test that the Ray server can handle a name query."""
    response = requests.post(
        ontical_model_server_url,
        json=[bob_thread_ids[0], "What is your name?"],
    )
    assert (
        response.status_code == 200
    ), f"Server returned {response.status_code}: {response.text}"
    result = response.json()
    assert isinstance(result, list), f"Result {result}"
    assert len(result) == 2, f"Result {result}"
    text_response, structured_response = result
    assert text_response
    response_data = NameAgeOccupation(**structured_response)
    assert response_data.name == "Bob"


def test_age_query_ray_server_call(
    ontical_model_server_url: str, bob_thread_ids: list[str]
):
    """Test that the Ray server can handle an age query."""
    response = requests.post(
        ontical_model_server_url,
        json=[bob_thread_ids[1], "How old are you?"],
    )
    assert (
        response.status_code == 200
    ), f"Server returned {response.status_code}: {response.text}"
    result = response.json()
    assert isinstance(result, list), f"Result {result}"
    assert len(result) == 2, f"Result {result}"
    text_response, structured_response = result
    assert text_response
    response_data = NameAgeOccupation(**structured_response)
    assert response_data.age == 28


def test_occupation_query_ray_server_call(
    ontical_model_server_url: str, bob_thread_ids: list[str]
):
    """Test that the Ray server can handle an occupation query."""
    response = requests.post(
        ontical_model_server_url,
        json=[bob_thread_ids[2], "What is your occupation?"],
    )
    assert (
        response.status_code == 200
    ), f"Server returned {response.status_code}: {response.text}"
    result = response.json()
    assert isinstance(result, list), f"Result {result}"
    assert len(result) == 2, f"Result {result}"
    text_response, structured_response = result
    assert text_response
    response_data = NameAgeOccupation(**structured_response)
    assert response_data.occupation.lower() == "bartender"

import requests

from test.schemas import Colors


def test_flag_color_ray_server_call(ontical_model_server_url: str):
    response = requests.post(
        ontical_model_server_url,
        json=["B", "What are the colors in the American flag?"],
    )
    assert (
        response.status_code == 200
    ), f"Server returned {response.status_code}: {response.text}"
    result = response.json()
    assert isinstance(result, list), f"Result {result}"
    assert len(result) == 2, f"Result {result}"
    text_response, structured_response = result
    assert text_response
    colors = Colors(**structured_response)
    assert colors.colors

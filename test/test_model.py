import requests


def test_flag_color(ontical_model_server_url: str):
    response = requests.post(
        ontical_model_server_url,
        json=["B", "What are the colors in the American flag?"],
    )
    assert (
        response.status_code == 200
    ), f"Server returned {response.status_code}: {response.text}"

    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 2

    reply = result[0]
    colors_dict = result[1]

    assert reply
    assert colors_dict
    assert "colors" in colors_dict
    assert len(colors_dict["colors"]) > 0

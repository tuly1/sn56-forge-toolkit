from forge.telemetry import _sanitize_value


def test_telemetry_redacts_nested_credential_keys():
    value = {
        "api_key": "api-private",
        "nested": {
            "accessKey": "access-private",
            "session_id": "session-private",
            "phase": "load_model",
        },
    }

    assert _sanitize_value(value) == {
        "api_key": "<redacted>",
        "nested": {
            "accessKey": "<redacted>",
            "session_id": "<redacted>",
            "phase": "load_model",
        },
    }


def test_telemetry_strips_signed_urls_userinfo_and_inline_secrets():
    value = (
        "api_key=super-secret cookie:chocolate "
        "s3://user:password@private-bucket/model.safetensors?X-Amz-Signature=x"
    )

    sanitized = _sanitize_value(value, key="error")

    assert "super-secret" not in sanitized
    assert "chocolate" not in sanitized
    assert "user:password" not in sanitized
    assert "Signature=x" not in sanitized
    assert "s3://private-bucket/model.safetensors" in sanitized


def test_telemetry_keeps_nonsecret_token_counts():
    assert _sanitize_value(2048, key="tokenized") == 2048
    assert _sanitize_value(86, key="planned_steps") == 86


def test_telemetry_redacts_auth_cookie_and_common_token_forms():
    message = (
        "Authorization: Basic dXNlcjpwYXNz\n"
        "Cookie: session=one; csrf=two; preference=three\n"
        "client_secret=client-value refresh_token=refresh-value "
        "access_token=access-value AWS_SECRET_ACCESS_KEY=aws-value "
        "github_pat_1234567890abcdefghijklmnop "
        "hf_1234567890abcdefghijklmnop"
    )

    sanitized = _sanitize_value(message, key="tail")

    for secret in (
        "dXNlcjpwYXNz",
        "session=one",
        "csrf=two",
        "client-value",
        "refresh-value",
        "access-value",
        "aws-value",
        "github_pat_",
        "hf_1234567890",
    ):
        assert secret not in sanitized

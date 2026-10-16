"""统一异常与错误码。API 层把它翻译成结构化 JSON 响应。"""
from __future__ import annotations


class AppError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class InvalidInputError(AppError):
    code = "invalid_input"
    http_status = 400


class IndexNotReadyError(AppError):
    code = "index_not_ready"
    http_status = 409


class NotFoundError(AppError):
    code = "not_found"
    http_status = 404


class ProviderError(AppError):
    code = "provider_error"
    http_status = 502


class UnauthorizedError(AppError):
    code = "unauthorized"
    http_status = 401

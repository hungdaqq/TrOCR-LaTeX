# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: ocr_service.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'ocr_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11ocr_service.proto\"\x1b\n\nOcrRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\"\x1c\n\x0bOcrResponse\x12\r\n\x05latex\x18\x01 \x01(\t25\n\nOcrService\x12\'\n\nOcrPredict\x12\x0b.OcrRequest\x1a\x0c.OcrResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ocr_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_OCRREQUEST']._serialized_start=21
  _globals['_OCRREQUEST']._serialized_end=48
  _globals['_OCRRESPONSE']._serialized_start=50
  _globals['_OCRRESPONSE']._serialized_end=78
  _globals['_OCRSERVICE']._serialized_start=80
  _globals['_OCRSERVICE']._serialized_end=133
# @@protoc_insertion_point(module_scope)

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""ResNeXt models for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import resnext

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.applications.resnext.ResNeXt50',
              'keras.applications.ResNeXt50')
@keras_modules_injection
def ResNeXt50(*args, **kwargs):
  return resnext.ResNeXt50(*args, **kwargs)


@keras_export('keras.applications.resnext.ResNeXt101',
              'keras.applications.ResNeXt101')
@keras_modules_injection
def ResNeXt101(*args, **kwargs):
  return resnext.ResNeXt101(*args, **kwargs)


@keras_export('keras.applications.resnext.decode_predictions')
@keras_modules_injection
def decode_predictions(*args, **kwargs):
  return resnext.decode_predictions(*args, **kwargs)


@keras_export('keras.applications.resnext.preprocess_input')
@keras_modules_injection
def preprocess_input(*args, **kwargs):
  return resnext.preprocess_input(*args, **kwargs)

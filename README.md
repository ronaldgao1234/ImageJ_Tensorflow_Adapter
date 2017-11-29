# Analysis in [ImageJ](https://imagej.net/)/[Fiji](http://fiji.sc) using [TensorFlow](https://www.tensorflow.org) models

Some experimentation with creating [ImageJ plugins](https://imagej.net/Writing_plugins)
that use [TensorFlow](https://www.tensorflow.org) image models.

For example, the one plugin right now pacakges the
[TensorFlow image recognition tutorial](https://www.tensorflow.org/tutorials/image_recognition),
in particular [its Java version](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java)
into a command plugin to label an opened image.

## How to use
This is the main source code but should use by installing Fiji app then updating the app with the plugin:
http://sites.imagej.net/Ronaldg/

```
This requires [Maven](https://maven.apache.org/install.html).  Typically `brew
install maven` on OS X, `apt-get install maven` on Ubuntu, or [detailed
instructions](https://maven.apache.org/install.html) otherwise.


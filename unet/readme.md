- I use unet plus plus rather than unet
  - longer training time is need.
  - more GPU RAM usage when training
  - but **higher performence** in image segmentation

- problem be resolve(don't know how to, and the side effect about it, need to do some research on google about it):
  - when in eval mode, want to just use layer 4(discard layer 1~3) o have less memory age nd less time need when doing forward propogation

from django import forms
from . models import ImageContents, ImageInference

class ImageContentsForm(forms.ModelForm):
    class Meta:
        model = ImageContents
        fields = ('image', )

class ImageInferenceForm(forms.ModelForm):
    class Meta:
        model = ImageInference
        fields = ('image_inf', )
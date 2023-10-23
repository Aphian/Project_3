from django import forms
from . models import ImageContents

class ImageContentsForm(forms.ModelForm):
    class Meta:
        model = ImageContents
        fidlds = ('image', )
from django import forms
class ImageUploadForm(forms.Form):   
    image = forms.ImageField(label='Select a file', help_text='max. 20 megabytes')

import requests

img = requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png").raw
with open('image.png', 'wb') as f:
    f.write(img)


    
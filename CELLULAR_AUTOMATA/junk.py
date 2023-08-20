import wget

URL = input('[RLE LINK] --> ')
fileName = input('[FILE NAME] --> ')

if fileName == '':
    response = wget.download(URL, 'SSV\RLEs')
else:
    response = wget.download(URL, 'SSV\RLEs\\'+fileName+'.rle')

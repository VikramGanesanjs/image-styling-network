# Image Styling Network


In an attempt to more deeply learn machine learning concepts, I built this machine learning model which can apply the style of one image on to another. It was trained on images of
art from different genres and styles, and the network learned how to extract a specific style from an image. Overall, this model was not very refined: most of the style that is pasted over
is usually the color values, and you only get some of the unique art style extraction. These image models often require long training times, and since this was a pet-project just to learn, 
I only trained it for a short time, about 5000 iterations. Overall, I learned a lot of the usual advanced math concepts used in machine learning during this project. It also really 
clicked with me how simple array operations like taking a standard deviation could get very complicated using tensors, and I had to learn how to implement these multi-dimensional operations,
even while using a machine-learning library like PyTorch. 

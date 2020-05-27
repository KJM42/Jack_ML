# Basics you need to start machine learning in Python
**J. Radford**
*(j.radford.1@research.gla.ac.uk)*

## *Contents:*
1. Zero to Hero quick setup *(for more step-by-step instructions with Python basics follow sections 2-5)*
2. Downloading software and handling virtual environments and packages with Anaconda
3. Getting to grips with Python using Spyder IDE
4. Setting up and testing Tensorflow and Keras
5. Intro to using Jupyter Notebook
6. Intro to using Google Colab

 ## 1. Zero to Hero quick setup
 
In a Nutshell, here are the commands you need to fast track your way from Zero to Hero and skip the explanations:
- **Open "Anaconda prompt"**
    - Type:
    
        - `conda create -n XtremeML python=3.7`
        
        - ``conda activate XtremeML``
        
        - ``conda install numpy matplotlib tensorflow keras spyder jupyter notebook``
        
        - ``spyder``
- **Now we've got Spyder open, check we have Tensorflow and Keras**
    - In the console (bottom right) type:
        - ```python
           import numpy as np
           ```
        - ```python
           import matplotlib.pyplot as plt
           ```
        - ```python
           import tensorflow as tf
           ```
        - ```python
           import keras as k
           ```
        - ```python
           tf.__version__  
           ```
         *(if this outputs a version number then we're good!)*
        
        - ```python
           (x_train, y_train),(x_test, y_test) = k.dataset.mninst.load_data()
            ```
        - ```python
           plt.imshow(x_train[0])
           ```
        
- **Try out jupyter notebook using Anaconda Navigator**
    - Open "Anaconda Navigator"
    
    - Use the drop down menu in the top left to go from **base** to **XtremeML**. Wait for it to load... It'll get there eventually.
    
    - You can check the packages in all your environments by going to "Environments" on the left panel. You can see all the packages installed but selecting the drop down menu and switching **installed** to **all** will also show you all available packages. You can also create, clone and delete environments here in the bottom left.
    
    - Head back to the home screen 
    
    - Now launch Jupyter Notebook
    
    - It opens a browser-based interface. Head to the folder you want to start a file in. *If you want to select a different drive then you'll need to close the window. Go to "Anaconda prompt", ``conda activate XtremeML`` change drive (for example E:/ drive, just type``e:``0. Now that we're in the new drive type ``jupyter notebook`` and the directory will open for the specified drive.*
    
    - Look for **New** in the top right and select **python 3**. Another window should open.
    
    - Do all the commands you did in Spyder (above) to check everything is working.
    
    - To run the cell simply hit "Run" in the command bar or Ctrl+Enter in windows. Hitting Shift+Enter will do the same but also create a new blank cell underneath.
    
    - To create some text go to the dropdown menu that says **Code** and select **Markdown**. This converts your cell into a text cell where you can write with Latex formatting and making headings etc. More info on markdown cells is here:
        
if you get an image then you're all set to do some machine learning!


- **Finally, if you want to try out Google Colab**
    - Go to: https://colab.research.google.com/
    
    - If you get a pop-up then you can open a recent document or hit **New Notebook** in the bottom right.
    
    - Alternatively, if you didn't get the pop-up you can go to **Files** in the top left and then choose **New Notebook**.
    
    - It works the same as jupyter notebook except everything is saved on google drive and the Markdown cells are a little easier to use.
    
    - Massive benefit is we now have access to whatever package we want because they are already installed by google. we can import things like **cv2** and **pytorch** without requiring any installation.
    
    - Another massive benefit is access to Google's servers and run your code on CPU, GPU and even a TPU without using your own computer. To activate a GPU or TPU (Tensor processing unit, see https://en.wikipedia.org/wiki/Tensor_processing_unit), go to the **Runtime** in the top left and choose **Change Runtime Type**.
    
    - You can download your code writtren in Colab as a `.py` files to run in Spyder or in `.ipyn` to run in Jupyter Notebook in the **File** menu in the top right.
    
    - Note you only have 12hr sessions on Google Colab so don't go running 3 day long codes, and you also have limited space and RAM depending on how busy the server is. Check the top right and hover your mouse over **RAM** for info. If instead you see **Connect** then click this because you're not connected to the server (you probably timed out).


*For a step by step walkthrough and some more background on using python then read below.*

# 2. Downloading software and handling virtual environments and packages with Anaconda

*Anaconda is a package manager that will install all the correct modules we need and their dependacies to run with good version control.*

- First go to: https://www.anaconda.com/products/individual

- Scroll to the bottom of the page and install the **python 3.7** version for your operating system

- Once it's downloaded just hit "*next, next, next, ...*" on the install wizard and you should be good to go!

### Open "Anaconda prompt"

Anaconda prompt is a terminal interface for Anaconda that is quick and easy to get packages we'll need.

First we want to start a virtual environment. Environments are good for version control. 
We install all the packages we need for a particular project in an "Environment" and then when we want to run our code again in 2050 when python and all our packages are updated to new versions, then we simply load up our environment and all the exact versions of the packages to run the code should still be installed. As if we had a time-machine in our computer to go back to wen the code worked. Everytime you start a new project it's good practice to start a new environment so that your new projects can use the most up-to-date packages without overwriting the package versions needed to run your old code.

- Create a new virtual environment by typing ``conda create -n XtremeML``, you can call it whatever you want but I called mine "XtremeML"
- Now we need to activate our new environment before installing things. Type ``conda activate XtremeML`` to change environments. You should see the name in the brackets change from ``(base)`` to ``XtremeML`` (or whatever you called yours). The "base" environment is your default environment. It's good to keep simple packages that are likely to be used in all your future environments like Numpy and Matplotlib and then we can simply clone that environment for future use instead of starting from scratch. 
- Now we need to install python. At the time of writing (26.05.20), we'll need python 3.7 to work with tensorflow. Type ``conda install python=3.7``.
- To begin coding, we need a nice Integrated Development Environment (IDE). I like Spyder, lets install it with ``conda install Spyder``

# 3. Getting to grips with Python using Spyder IDE

- Open spyder by literally just typing ``spyder`` into the Anaconda prompt terminal
- Now a program should open which we can use to start coding in python.
- *Packages*, are folders containing *modules* which have useful pre-written *functions* that let us code in python easily. To do any sort of mathematical manipulation of matrices then we'll need the module Numpy.
- In the terminal (on the bottom right of the screen) type ``import numpy``
- We should get an error, that's because we're haven't installed it yet!
- Close spyder

### Open "Anaconda prompt"

 - Make sure Spyder is closed.
 - Go back to our Anaconda prompt terminal and type ``conda install numpy``
 - Now let check if it's working. This time, instead of opening spyder from this terminal, lets have a look at "Anaconda Navigator".
 
### Open "Anaconda Navigator"
 
 - Open Anaconda Navigator GUI, it's already installed on your computer.
 - We should see a drop down menu in the top left which says **base**. Click on the drop down menu and select **XremeML**
 - Now we wait for it to load.. it may take around 30secs to change environment. *(note: this is usually why using Anaconda prompt is better when possible!)*
 - You should now notice that Spyder is already installed. Click "launch" to open Spyder.
 
### Back to Spyder
 
 - Now try ``import numpy`` again in the Spyder console. Hopefully there is no error message this time!
 - Here's some Python basics, try typing:
 ```python
 a = 1
```

 Above the console there is a "Help" window with a big box saying "Usage", it's not that useful, click the tab which says "Variable explorer". We should see our newly defined integer there.
 ```python
 b = [1,2,3]
 ``` 
Now we've created a *list*, it will also show up in the variables window and we can even see the contents! Lets try and access the 1st element of the list. Usually in MatLab format we would type: 
```python
b(1)
```
Unfortunately, we got an error, this is because using curved brackets is only for use when calling a function. We'll need square brackets
```python
b[1]
```
This time, no error, but notice we got the wrong value! The first element of `b` is `1` not `2`. This is because python indexing begins with the 0th element, so the indexes for b are actually 0,1,2.
```python
b[0]
```
Now it should ive us the correct element. Lets try outputing some string, instead of `disp('hello world')` in Matlab we'll need:
```python
print('Hello world')
```
One last difference from matlab is taking the power of a number, we can't use `2^3` because this symbol is used for a logic operator in python we need a double asterisk:
```python
2**3
```

### Lets try a sin plot

```python
x = numpy.linspace(0, 2*np.pi, 1000)
y = numpy.sin(x)
```

Now to plot our sine values in a figure, we'll need the plotting module **matplotlib.pyplot**. We haven't installed this yet so head back to your Anaconda prompt.

It's annoying to always write **numpy**, lets use an abbreviation by typing
```python
import numpy as np
```

### Back to Anaconda prompt

 - Hopefully you haven't closed this window and you opened Spyder using Anaconda Navigator. If not then re-open Anaconada prompt and type ``conda activate XtremeML`` to get back into our environment.
 - Install matplotlib with ``conda install matplotlib``
 
### Back to Spyder
```python
import matplotlib.pyplot as plt
```
Now we can plot a figure using
``` python
plt.plot(x,y)
```
 - Annoyingly, the latest Spyder puts our plots into a plots window. Find the tab "plots" along from the "Variable explorer". For interactive plots, find the **little spanner symbol** near the top of the screen. This is your **preferences**, go to **iPython console**, then go to the **Graphics** tab. Here we need to select **Backend: Automatic**. It should already be selected but just select it again to make sure. Hit **Ok** and restart Spyder using Anaconda Navigator like before (don't use the Anaconda prompt, we'll need it again soon!).

 - Frustratingly now we need to write everything out again, including the imports. Make your life easier by using the editor on the left of the screen. You're currently seeing a temporary script, don't bother using this, it's a default script which always opens and is sometimes used to test out code before properly writing a proper file, I don't really see the benefit. 
  - Begin a new script by clicking the **new page symbol** in the top left, it's the same as MS Word, you'll know what I mean when you see it.
  - First thing to do is hit save, choose your directory and give your script a name.
  - Now lets plot our sine wave again.
```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 1000) # create a vector of angles
y = np.sin(x)                     # find the sine values
plt.plot(x, y)                    # plot the angles vs. sine
```
- Hit **F5** or the **play button** at the top of the screen to run the script in the console.
- Now we should see a nice interactive plot in a separate figure window
- Always comment code using the `#` to make it readable to others





- Lets try overlaying the plot with various sine waves with different phase using a for loop and a defining a function
```python
def my_func(x, p):
    # Calculate the sine wave for 
    # angles (x) and phase (p)
    
    y = np.sin(x) + p
    return y
```
This is how to define a function, take care with the `:` and the indents. Everything after the `:` that is indented will be included in the function. The `return` will define what is given as the output of the function, this can be multiple objects, just separate them with a comma.

- Continue writing our script with:
```python
x = np.linspace(0, 2*np.pi, 1000)
for i in range(5):
    p = i*0.1                       # change the phase offest value every iteration
    y = my_func(x, p)               # call our function to get y values
    plt.plot(x, y, label='i={}'.format(i))
plt.legend()
plt.title('My plot') 
```
`x` stays the same, so it's outside of the `for` loop. The `for` loop works the same as the function. Everything indented after the `:` will be included. The `range` function will produce a set of numbers starting from zero with the length defined in the argument. So here, we will iterate over 0,1,2,3,4 but not 5! Always think of the range argument as the number of iterations you want, but note that `i` will take the integer values **up to, but not including** this number. Notice The legend and title is plotted after the for loop is finished since it's not indented. Python is very sensitive to indents for this reason, if you ever get an indent error then you've forgotten to indent or there is too much indent. Indent using the Tab key, unindent using Shift+Tab. The `label` in the plot function, is a keyword argument. This will be used for the legend, to name the independent sine graphs. This also introduces the `.format()` function, use `{}` wherever the value should be in the string, and put the variable name as the argument. 

- Lets run again by using the **F5** key.

- Finally, to save and load numpy arrays in the same directory as our code, try this in the console:
``` python
np.savez('x_values',x)
```
This function will save our vector of angles as a zipped numpy array called "x_values.npz". To access it again we need:
```python
angles = np.load('x_values.npz')
```
Unfortunately, because we zipped the data, we get a `NpzFile Object` for `angles` and not the numpy array that we expected. To have a look at what is stored inside this object we can interrogate it using:
```python
angles.files
```
This will show us that there is a numpy array called `arr_0` within the object. This is the default name always given to an array that is saved using the `np.savez()` function. To access the array we need to use the command:
```python
angles_array = angles['arr_0']
```
Now we have the numpy array with the contents of the previously defined `x`, but with the name `angles_array`. To save time we can also use this square brackets argument when loading the data in the equivalent expression:
```python
angles_array = np.load('x_values.npz')['arr_0']
```
If we don't want the default name then we can choose a different name in the saving process using a keyword argument:
```python
np.savez('x_values', angles=x)
```
Now the complimentary load command would be:
```python
angles_array = np.load('x_value.npz')['angles']
```

# 4. Setting up and testing Tensorflow and Keras

- Just like the matplotlib package is used for plotting graphs, we need to get a package (or an Application Programming Interface (API)) to do the underlying hardware processes we need to train neural networks. This is Tensorflow API, it's developed by Google but a occasionally still cumbersome to work with. 
- Tensorflow is made easier using the Keras API which allows  us to code with minimal effort and outputs much easier to interpret error messages.

### Back to Anaconda prompt
- Again, make sure you're still in the **XtremeML** environment.
- Then, you guessed it: ``conda install tensorflow keras``

### Back to Spyder
- Lets make sure they installed properly, using the console type:
    ```python
    import tensorflow as tf
    import keras as k
    ```

    Now check the tensorflow version:
    ```python
    tf.__version__
    ```
    It should output the version that was installed. It might be a version behind the latest but this is just because anaconda lags a little behind the tensorflow updates.

    Try importing the MNIST dataset using Keras:
    ```python
    (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
    ```
    For context, `x_train` is the training images of handwritten numbers and `y_train` is there corresponding labels which tells which number is represented in the image. Likewise, `x_test` and `y_test` is some unseen images and labels used to test the performance of an algorithm after it has trained on the training data. Try looking at the shape of the x_train data set:

    ```python
    x_train.shape
    ```
    We see from the output that the x_train data is made up of $60,000$ images of $(28\times 28)$ resolution. Lets see the first image in the training dataset:

    ```python
    plt.imshow(x_train[0])
    ```

    The output of this keras function is two *tuples*. A tuple has curved brackets and cannot be altered like a list can. we can still access elements of a tuple using the square bracket indexing but we can't rearrange values or perform mathematical operations on them. In this case, we've specifically defined what is expected to be inside these two tuples that Keras gave us, so python simply saves each array individual to avoid dealing with tuples.

- To demonstrate tuples try:
```python
train, test = k.datasets.mnist.load.data()
```
Now `train` and `test` are tuples, looking in the variable explorer, we can see that each tuple contains the data for both images and labels. To unpack the training images `x_train` from the `train` tuple we can just use:
```python
x_train = train[0]
```

- Dictionaries are another structure which is useful for storing data under one structure. Dictionaries have *keys* which are strings used to access particular *values*. To stick with our example of the training and test data of MNIST we already have, a dictionary which keeps all of these arrays under one structure that can be accessed by calling it's name with a key is made by:
    ```python
    mnist = {'training_images':x_train, 
             'training_labels':y_train, 
             'test_images':x_test, 
             'test_labels':y_test}
    ```
    We can now have a look at all the things stored in our dictionary using:
    ```python
    mnist.keys()
    ```
    
    If we wanted to then access the test images:
    ```python
    mnist['test_images']
    ```

# 5. Intro to Jupyter Notebook

So far we haven't installed Jupyter Notebook so lets start from the Anaconda Navigator. Jupyter Notebook is a browser app as an interface but it doesn't need an internet connection. The advantages are to run code in cells with accompanying explanations which can even be used to produce a PDFs. 

### Back to Anaconda Navigator

- Remember to switch environment to XtremeML using the drop down menu if you're opening Navigator up again.
- Click install next to Jupyter Notebook. *Note: Jupyter Lab is **not** the same, this is another IDE which is a mix of Spyder and Jupyter notebook styles. Try it out later if you want!*
- Once it's installed click launch
- You'll see a browser window with your directories. Choose a directory you want to save a new file in and click **New** in the top right corner, and select **Python 3** to make a new notebook file in this directory.
- A new window should open.
- Try typing in the first cell:
    ```python
    import matplotlib.pyplot as plt
    import keras as k 
    ```
    We can run this cell a number of ways, Ctrl+Enter will only run the cell, Shift+Enter will run the cell and open a new cell underneath. Alternatively, you can click the **Run** button on the top panel.
- In the second cell, lets import the fashion MNIST data set using keras and use a subplot to visualise the first 25 images of the training data.
    ```python
    (x_train, y_train),(x_test, y_test) = k.datasets.fashion_mnist.load_data()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(x_train[i])
        plt.axis('off')
        plt.title('{}'.format(i+1))
    plt.suptitle('fMNIST training set', size=20)
    ```
    Here I've used a subplot in the same way as MatLab, note that `plt.subplots` with an added "s", is a different object-orientated approach to subplotting. The subplot function cannot have a 0th plot so it begins at `i+1`. Notice that I've set a title to every image in the plot but to give the figure an overall heading, The function `plt.suptitle()` is used after the loop is finished.
    
- A huge benefit of Jupyter Notebooks is merging text descriptions with code. Create a new cell by running the most recent cell with Shift+Enter or clicking the **+** sign on the top panel next to the save button.
- Change the cell from **Code** to **Markdown** using the dropdown menu on the top panel.
- We can now type text in the cell and use Latex formatting for equations etc. More info on Markdown cells can be found here:https://medium.com/analytics-vidhya/the-ultimate-markdown-guide-for-jupyter-notebook-d5e5abf728fd
- This document you're reading was made in Jupyter Notebook!




# 6. Google Colab
Forget installing software, no need to buy crazy gaming computers. If all you want to do is code then use Google Colab. It's a completely online tool using Google servers to run your code. Colab already has everything you need already installed so just import anything you like. It also allows you to run code using GPUs and TPUs (Tensor Processing Units, hardware specifically designed for neural networks - https://en.wikipedia.org/wiki/Tensor_processing_unit).

### Go to https://colab.research.google.com/

- You'll need a google account to access Colab so sign up or log in if you haven't already
- You might see a pop-up window which has recent activity if you've visited the site before, follow the instrucitons or hit cancel.
- If you didn't see a pop-up, or closed it by mistake, go to **Files** in the top left and click **New notebook**.
- You'll see something similar to Jupyter Notebook with some subtle differences.
- In the first cell:
    ```python
    import cv2
    import numpy as np
    ```
- If you typed this out, you'll notice that Colab has autocomplete. This is super useful for learning to use new modules and for finding new functions.
- Now we can use the same keyboard shortcuts to run or hit the **play button** on left of the cell.
- Markdown cells are a little easier too, click on **+ Text** in the top left to start a Markdown cell. There is an easy toolbar which will automatically fill out typesetting code.
- Lets try using a GPU, make anew code cell using **+ Code** in the top left and type:
    ```python
    import tensorflow as tf
    tf.config.list_physical_devices()
    ```
    This will output the hardware in the Google server that our kernel is currently using to run code. To switch to GPU go to **Runtime** on the top panel and select **Runtime Type**. Now we have the dropdown menu for **Hardare Accelerator** with the options of **GPU** and **TPU**. Select **GPU** and hit **Save**.
- Before using the GPU, all the cells need to be re-run. Do this by oing to **Runtime** again and selecting **Run all**.
- Now we should see the most recent cell output the GPU as a physical device.

- All the notebooks will be saved to Google Drive but to sotre them offline go to **File** in the top left and you'll see options to download the `.ipyn` notebook script which can be run offline in Jupyter Notebook or the `.py` files which can be run in Spyder. 

- You can access Google Colab from any device that has internet and the memory allowance will vary depending on the activity of the servers when you start a session. There are other limitations to using Google Colab, there is a really useful FAQ section to help you determine whether using Colab is a good idea for you: https://research.google.com/colaboratory/faq.html 

    

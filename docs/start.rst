Installation
============

.. note:: Since the API changes with each release,you may want to check the `CHANGELOG.md <https://github.com/pygfx/wgpu-py/blob/main/CHANGELOG.md>`_ when you upgrade to a newer version of wgpu.

Install with pip
----------------

You can install ``wgpu-py`` with your favourite package manager (we use ``pip`` in the example commands below).
Python 3.9 or higher is required. Pypy is supported.

.. code-block:: bash

    pip install wgpu


Since most users will want to render something to screen, we recommend installing GLFW as well:

.. code-block:: bash

    pip install wgpu glfw


GUI libraries
-------------

Multiple GUI backends are supported, see :doc:`the GUI API <gui>` for details:

* `glfw <https://github.com/FlorianRhiem/pyGLFW>`_: a lightweight GUI for the desktop
* `jupyter_rfb <https://jupyter-rfb.readthedocs.io>`_: only needed if you plan on using wgpu in Jupyter
* qt (PySide6, PyQt6, PySide2, PyQt5)
* wx


The wgpu-native library
-----------------------

The wheels that pip installs include the prebuilt binaries of `wgpu-native <https://github.com/gfx-rs/wgpu-native>`_, so on most systems everything Just Works.

On Linux you need at least **pip >= 20.3**, and a recent Linux distribution, otherwise the binaries will not be available. See below for details.

If you need/want, you can also `build wgpu-native yourself <https://github.com/gfx-rs/wgpu-native/wiki/Getting-Started>`_.
You will then need to set the environment variable ``WGPU_LIB_PATH`` to let wgpu-py know where the DLL is located.


Platform requirements
---------------------

Under the hood, wgpu runs on Vulkan, Metal, or DX12. The wgpu-backend
is selected automatically, but can be overridden by setting the
``WGPU_BACKEND_TYPE`` environment variable to "Vulkan", "Metal", "D3D12",
or "OpenGL".

Windows
+++++++

On Windows 10+, things should just work. If your machine has a dedicated GPU,
you may want to update to the latest (Nvidia or AMD) drivers.

MacOS
+++++

On MacOS you need at least 10.13 (High Sierra) to have Metal/Vulkan support.

Linux
+++++

On Linux, it's advisable to install the proprietary drivers of your GPU (if you
have a dedicated GPU). You may need to ``apt install mesa-vulkan-drivers``. On
Wayland, wgpu-py requires XWayland (available by default on most distributions).

Note that WSL is currently not supported.

Binary wheels for Linux are only available for **manylinux_2_24**.
This means that the installation requires ``pip >= 20.3``, and you need
a recent Linux distribution, listed `here <https://github.com/pypa/manylinux#manylinux>`_.

If you wish to work with an older distribution, you will have to build
wgpu-native yourself, see "dependencies" above. Note that wgpu-native
still needs Vulkan support and may not compile / work on older
distributions.

Cloud Compute
+++++++++++++

GPU Environments
^^^^^^^^^^^^^^^^

WGPU can work in GPU cloud compute environments on Linux machines with no
physical display output. By default, these environments may lack system
libraries that are typically found on a standard linux desktop. On Debian &
Ubuntu based systems you should be able to get everything you need by installing
the following in addition to your vendor-specific (Nvidia/AMD) GPU drivers:

.. code-block:: bash

    sudo apt install xserver-xorg-core mesa-vulkan-drivers libvulkan1

.. note:: If your distro is not Debian/Ubuntu install the corresponding packages for your distribution.

You can verify whether the `"DiscreteGPU"` adapters are found:

.. code-block:: python

    import wgpu
    import pprint

    for a in wgpu.gpu.enumerate_adapters_sync():
        pprint.pprint(a.info)

If you are using a remote frame buffer via `jupyter-rfb <https://github.com/vispy/jupyter_rfb>`_ we also recommend installing the following for optimal performance:

.. code-block:: bash

    sudo apt install libjpeg-turbo8-dev libturbojpeg0-dev
    pip install simplejpeg

Your mileage may vary across different cloud service providers, for more info see: https://github.com/pygfx/wgpu-py/issues/493

Installing LavaPipe on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run wgpu on systems that do not have a GPU (e.g. CI) you need a software renderer.
On Windows this (probably) just works via DX12. On Linux you can use LavaPipe:

.. code-block:: bash

        sudo apt install libegl1-mesa-dev libgl1-mesa-dri libxcb-xfixes0-dev mesa-vulkan-drivers

.. note::

    The precise visual output may differ between different implementations of Vulkan/Metal/DX12.
    Therefore you should probably avoid per-pixel comparisons when multiple different systems are
    involved. In wgpu-py and pygfx we have solved this by generating all reference images on CI (with Lavapipe).

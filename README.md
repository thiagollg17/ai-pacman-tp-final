# ai-pacman-tp-final

**⚠️ ACLARACIONES PARA MANEJAR EL REPOSITORIO**  

Esto incluye instrucciones sobre cómo organizar las carpetas y archivos y ejecutar el código, tanto de manera secuencial como en paralelo en Google Colab.  

**Instrucciones generales**   
1. Carpeta images: incluir la carpeta images en el entorno de ejecución, ya que contiene los recursos gráficos necesarios para visualizar el juego.  

**Entorno local pacman secuenciales (con mating pool)**  
1. Tener la carpeta images en la misma ubicación que los archivos pacman.py y pacai.py  
2. Ejecuta pacman.py, el cual a su vez llamará a pacai.py.  
Este modo ejecutará solo un agente a la vez y no pasará al siguiente individuo de la generación hasta que el actual termine.  

**Ejecucion en Google Colab paralelo (con mating pool)**  
En la carpeta colab encontrarás los archivos necesarios para ejecutar todos los individuos de una generación en paralelo.  

1. Copia los archivos pacman.py y pacai.py en la carpeta content en Colab.  
2. En la carpeta content, crea una subcarpeta llamada images y replica la estructura de la carpeta images original y sube todos los archivos de images por carpeta(directorio).  

<div align="center">
   <img src="/readme1imagen.png" alt="imagen1estructuracontent">
</div>

<div align="center">
   <img src="/readme2imagen.png" alt="imagen2estructuracontent">
</div>



En Colab, puedes ejecutar todos los individuos de la generación en paralelo, con un máximo recomendado de 120, o preferiblemente de 100 o menos, caso contrario se cuelga en varias ocasiones.  

3. En Colab, no cambies los nombres de los archivos ni la estructura de carpetas, ya que pacman.py hace un import pacai as ga, por lo cual si se cambia el nombre a pacai y pacman por conveniencia porfavor cambiar los imports de los archivos asi se ejecuta de forma correcta.  
4. Para iniciar la ejecución, utiliza el comando: !python pacai.py. Ahora pacman.py solo calcula el fitness y se lo pasa al pacai.py el cual ahora tiene toda la logica del mating pool que anteriormente no era asi.

**Ejecucion en Google Colab paralelo (con rejection sampling)**  
1. Carpeta rejection_sample: Esta carpeta contiene una implementación que usa el método de rejection sampling. En esta versión, solo consta del archivo pacai.py ya que se utiliza el mismo archivo pacman.py de la carpeta colab (el anteriormente mencionado para colab con mating pool)  
2. En colab no reemplaze el pacman.py ya que se usa el mismo que el de mating pool del anterior punto. Solo reemplaze el pacai por el de esta carpeta. Ejecute !python pacai.py. Nuevamente la logica del metodo esta en el pacai y no en el pacman.py (este calcula solo fitness)
3. Recuerde no cambiar el nombre de los archivos. Si lo quiere hacer por conveniencia cambie los imports internos asi coinciden.

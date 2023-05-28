import cv2

# Iniciar la cámara
cap = cv2.VideoCapture(0)

# Obtener el primer fotograma
ret, frame = cap.read()
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    # Leer el fotograma actual
    ret, frame = cap.read()

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular la diferencia entre el fotograma actual y el anterior
    frame_diff = cv2.absdiff(gray, prev_frame)

    # Aplicar un umbral para resaltar los cambios significativos
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Aplicar operaciones de morfología para eliminar el ruido
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, None, iterations=2)
    threshold = cv2.dilate(threshold, None, iterations=2)

    # Encontrar los contornos de las regiones con cambios
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en el fotograma original
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Actualizar el fotograma anterior
    prev_frame = gray

    # Mostrar la imagen resultante
    cv2.imshow('Detección de movimiento', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()

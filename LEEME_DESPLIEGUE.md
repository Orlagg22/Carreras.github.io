# Publicar este juego en tu GitHub Pages

1. Crea un repositorio vacío en tu GitHub (por ejemplo `noryx-web`), sin README.
2. Sube este contenido:
   ```bash
   git init -b main
   git add .
   git commit -m "Noryx web"
   git remote add origin https://github.com/TU_USUARIO/noryx-web.git
   git push -u origin main
   ```
3. En GitHub: **Settings → Pages → Source: GitHub Actions**.
4. Espera a que termine la acción *Deploy to GitHub Pages* (pestaña Actions, ~3-5 min).
5. Juega en `https://TU_USUARIO.github.io/noryx-web/play`

## Probar en tu PC
```bash
npm ci
npm run dev        # http://localhost:3000
```
No abras los archivos con doble clic (`file://`): el juego usa workers y necesita un servidor web.

## Si dice que se detuvo por gráficos
Activa la aceleración por hardware del navegador (Chrome/Edge: Configuración → Sistema), actualiza el driver de video y recarga. El juego también funciona con render por software, pero lento.

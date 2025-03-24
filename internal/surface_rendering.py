from skybox_utils import obtain_dirs_for_skybox


class SurfaceRenderer:
    def __init__(self, height=256):
        self.env_dirs = obtain_dirs_for_skybox(height=height)
    
    def render_irradiance(self, normal, env_map):
        pass

    def render(self, 
        normal, 
        env_map,
        albedo,
        roughness,
        metallic,
        ):

        pass

    def render_diffuse(self,):
        pass

    def render_specular(self,):
        pass
//
// Implementation for Yocto/RayTrace.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "yocto_raytrace.h"

#include <yocto/yocto_cli.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_shading.h>
#include <yocto/yocto_shape.h>

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR SCENE EVALUATION
// -----------------------------------------------------------------------------
namespace yocto {

// Generates a ray from a camera for yimg::image plane coordinate uv and
// the lens coordinates luv.
static ray3f eval_camera(const camera_data& camera, const vec2f& uv) {
  auto film = camera.aspect >= 1
                  ? vec2f{camera.film, camera.film / camera.aspect}
                  : vec2f{camera.film * camera.aspect, camera.film};
  auto q    = transform_point(camera.frame,
      {film.x * (0.5f - uv.x), film.y * (uv.y - 0.5f), camera.lens});
  auto e    = transform_point(camera.frame, {0, 0, 0});
  return {e, normalize(e - q)};
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace yocto {

// Raytrace renderer.
static vec4f shade_raytrace(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return rgb_to_rgba(eval_environment(scene, ray.d));
  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];
  auto& material = scene.materials[instance.material];  // eval material
  auto  outgoing = -ray.d;

  auto normal = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));
  auto position = transform_point(
      instance.frame, eval_position(shape, isec.element, isec.uv));
  vec2f rg        = eval_texcoord(shape, isec.element, isec.uv);
  auto  textcoord = vec2f{fmod(rg.x, 1), fmod(rg.y, 1)};

  // color
  auto roughness = material.roughness;
  auto color     = material.color;
  if (material.color_tex >= 0)
    color *= rgba_to_rgb(
        eval_texture(scene.textures[material.color_tex], textcoord, true));
  if (material.normal_tex >= 0)
    color *= rgba_to_rgb(
        eval_texture(scene.textures[material.normal_tex], textcoord, true));

  if (material.emission_tex >= 0)
    color *= rgba_to_rgb(
        eval_texture(scene.textures[material.emission_tex], textcoord, true));

  if (material.roughness_tex >= 0)
    color *= rgba_to_rgb(
        eval_texture(scene.textures[material.roughness], textcoord, true));

  if (material.scattering_tex >= 0)
    color *= rgba_to_rgb(
        eval_texture(scene.textures[material.scattering_tex], textcoord, true));

  // Opacity
  if (rand1f(rng) <
      1 - eval_material(scene, instance, isec.element, isec.uv).opacity)
    return shade_raytrace(
        scene, bvh, ray3f{position, ray.d}, bounce + 1, rng, params);

  // Radiance
  auto radiance = rgb_to_rgba(material.emission);

  if (bounce >= params.bounces) return radiance;

  // hair
  if (!shape.points.empty())
    normal = -ray.d;
  else if (!shape.lines.empty())
    normal = orthonormalize(-ray.d, normal);
  else if (!shape.triangles.empty())
    if (dot(-ray.d, normal) < 0) normal = -normal;

  // Materials
  switch (material.type) {
    case material_type::matte: {
      auto incoming = sample_hemisphere_cos(normal, rand2f(rng));
      radiance += rgb_to_rgba(color) * shade_raytrace(scene, bvh,
                                           ray3f{position, incoming},
                                           bounce + 1, rng, params);
      break;
    }
    case material_type::reflective: {
      if (material.roughness == 0) {
        auto incoming = reflect(outgoing, normal);
        radiance += rgb_to_rgba(
            fresnel_schlick(color, normal, outgoing) *
            rgba_to_rgb(shade_raytrace(scene, bvh, ray3f{position, incoming},
                bounce + 1, rng, params)));
      } else {
        auto exponent = 2 / (pow(material.roughness, 4));
        auto halfway  = sample_hemisphere_cospower(
            exponent, normal, rand2f(rng));
        auto incoming = reflect(outgoing, halfway);
        radiance += rgb_to_rgba(
            fresnel_schlick(color, halfway, outgoing) *
            rgba_to_rgb(shade_raytrace(scene, bvh, ray3f{position, incoming},
                bounce + 1, rng, params)));
      }
      break;
    }
    case material_type::glossy: {
      auto exponent = 2 / (pow(material.roughness, 4));
      auto halfway  = sample_hemisphere_cospower(exponent, normal, rand2f(rng));
      if (rand1f(rng) <
          fresnel_schlick({0.04, 0.04, 0.04}, halfway, outgoing).x) {
        auto incoming = reflect(outgoing, halfway);
        radiance += shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
      } else {
        auto incoming = sample_hemisphere_cos(normal, rand2f(rng));
        radiance += rgb_to_rgba(color) * shade_raytrace(scene, bvh,
                                             ray3f{position, incoming},
                                             bounce + 1, rng, params);
      }
      break;
    }
    case material_type::transparent: {
      auto test = fresnel_schlick({0.04, 0.04, 0.04}, normal, ray.d);
      if (rand1f(rng) < test.x) {
        auto incoming = reflect(outgoing, normal);
        radiance += shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
      } else {
        auto incoming = ray.d;
        radiance += rgb_to_rgba(color) * shade_raytrace(scene, bvh,
                                             ray3f{position, incoming},
                                             bounce + 1, rng, params);
      }
      break;
    }
    case material_type::refractive: {
      auto test = fresnel_schlick({0.04, 0.04, 0.04}, normal, ray.d);
      auto ior  = material.ior;
      if (rand1f(rng) < test.x) {
        auto incoming = reflect(outgoing, normal);
        radiance += shade_raytrace(
            scene, bvh, ray3f{position, incoming}, bounce + 1, rng, params);
      } else {
        if (dot(ray.d, normal) > 0)
          normal = -normal;
        else
          ior = 1 / ior;
        auto incoming = refract(outgoing, normal, ior);
        radiance += rgb_to_rgba(color) * shade_raytrace(scene, bvh,
                                             ray3f{position, incoming},
                                             bounce + 1, rng, params);
      }
      break;
    }
    case material_type::volumetric: {
      auto newcolor = one4f;
      auto newray   = ray3f{position, ray.d};
      bool in       = true;
      while (in) {
        if (bounce >= params.bounces) return zero4f;  // o radiance
        auto newisec = intersect_bvh(bvh, scene, newray);
        if (!newisec.hit)return rgb_to_rgba(eval_environment(scene, newray.d));


        auto maxdist = newisec.distance;
        auto hit     = -log(1 - rand1f(rng)) / material.ior;

        if (hit < maxdist) {
          auto newpos       = ray_point(newray, hit);
          auto newdirection = sample_sphere(rand2f(rng));  // H.W. distribuition

          newray = ray3f{newpos, newdirection};
          newcolor *= rgb_to_rgba(color);
          // radiance += newcolor; ha senso

        } else {
          if (newisec.instance != isec.instance) {
            auto& newinstance = scene.instances[newisec.instance];
            auto& newshape    = scene.shapes[newinstance.shape];
            auto& newmaterial = scene.materials[newinstance.material];

            auto newnormal   = transform_direction(newinstance.frame,
                eval_normal(newshape, newisec.element, newisec.uv));
            auto newposition = transform_point(newinstance.frame,
                eval_position(newshape, newisec.element, newisec.uv));

            auto incoming = sample_hemisphere_cos(newnormal, rand2f(rng));
            newray        = ray3f{newposition, incoming};
            newcolor *= rgb_to_rgba(newmaterial.color);
          } else {
            auto newpos = ray_point(newray, maxdist);
            newray      = ray3f{newpos, newray.d};
            radiance += newcolor * shade_raytrace(scene, bvh, newray,
                                       bounce + 1, rng, params);
            in = false;
          }
        }
      }

      break;
    }
  };
  return radiance;
}

static vec4f raymarch(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params,bvh_intersection isec,material_data material) {
    vec4f totalcolor = zero4f;
    double totalopacity = 0.0;
    auto dL = 0.01;
    auto passo = 0.0;

    while (passo <= isec.distance)  // finchè sto dentro l'oggetto minore uguale?
    {
     // printf("qui %lf %lf\n", passo, isec.distance);
      auto position = ray_point(ray, passo); //possibile implementazione migliore
    for (auto lights : scene.instances) {
       auto& light = scene.materials[lights.material];
      if (sum(light.emission) > 0) {
      auto lightpos =lights.frame.o;
        auto lightdir = normalize(lightpos-position);
        auto raytolight = ray3f{position, lightdir};
        auto newisec    = intersect_bvh(bvh, scene, raytolight);
       // auto density    = 1 / newisec.distance;
        auto passolight = 0.0;
        while (passolight <= newisec.distance) {
          //    printf("qui %lf %lf\n", material.opacity, newisec.distance);
          auto    newpos = ray_point(raytolight, passolight); 
          totalcolor += rgb_to_rgba(material.color) * (1.0 - totalopacity);
          totalopacity += (1.0 - totalopacity) * material.opacity;
          passolight += dL;
        }
        
      
      }
    
    }

    passo += dL;
  
   }
    return totalcolor;

}



// Matte renderer.
static vec4f shade_matte(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  // YOUR CODE GOES HERE ----
  return {0, 0, 0, 0};
}

//cell shading
static vec4f cell_shading(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng, const raytrace_params& params) {
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return zero4f;
  auto& instance = scene.instances[isec.instance];
  //auto& material = scene.materials[instance.material];
  auto  material = eval_material(scene, instance, isec.element, isec.uv);
  auto& shape    = scene.shapes[instance.shape];
  auto  position = transform_point(
      instance.frame, eval_position(shape, isec.element, isec.uv));
  auto  color    = material.color;
  auto  light    = vec3f { 0.4, 0.8, 0.8 };  
  auto  normal   = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));
  auto  NdotL           = dot(light, normal);
  auto light_intensity = smoothstep(0.0f, 0.01f, NdotL); //*light intensity
                      
  auto ambientColor = eval_environment(scene, ray.d);

  //specular reflection
  float glossiness = 32.0;
  auto  specCol    = vec3f{0.9, 0.9, 0.9};

  auto  viewDir    = normalize(-ray.d);
  auto halfVec = normalize(light+viewDir);
  auto NdotH   = dot(normal, halfVec);
  
  auto specularIntensity = smoothstep(0.005f,0.01f,pow(NdotH * light_intensity,glossiness*glossiness));
  auto specular = specularIntensity * specCol;

  //rim

  auto rimCol = one3f;
  auto rimAmount = 0.716f;
  auto rimThreshold = 0.1f;

  auto rim = material.type!= material_type::matte? smoothstep(rimAmount - 0.01f, rimAmount + 0.01f,( 1 - dot(viewDir, normal))*pow(NdotL,rimThreshold))*rimCol:zero3f;

  //shadow 
  auto shadowhit = intersect_bvh(bvh, scene, ray3f{position, light});
  if (shadowhit.hit) {
    if (scene.materials[scene.instances[shadowhit.instance].material]
            .emission == zero3f)
    color *= 0.1;
  }

  return rgb_to_rgba(
      color * (specular+light_intensity+ambientColor+rim)); 
}
    // Eyelight renderer.
static vec4f shade_eyelight(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return zero4f;
  auto& instance = scene.instances[isec.instance];
  auto& material = scene.materials[instance.material];
  auto& shape    = scene.shapes[instance.shape];
  auto  normal   = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));
  return rgb_to_rgba(material.color * dot(normal, -ray.d));
}

static vec4f shade_normal(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return zero4f;
  auto& instance = scene.instances[isec.instance];
  auto& shape    = scene.shapes[instance.shape];
  auto  normal   = transform_direction(
      instance.frame, eval_normal(shape, isec.element, isec.uv));
  normal = (normal * 0.5) + 0.5;
  return rgb_to_rgba(normal);
}

static vec4f shade_texcoord(const scene_data& scene, const bvh_scene& bvh,
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return zero4f;
  auto& instance  = scene.instances[isec.instance];
  auto& element   = isec.element;
  auto& shape     = scene.shapes[instance.shape];
  vec2f textcoord = eval_texcoord(shape, element, isec.uv);
  return {fmod(textcoord.x, 1), fmod(textcoord.y, 1), 0.0f, 0};
}

static vec4f shade_color(const scene_data& scene,
    const bvh_scene&                       bvh,  // cambiare instance
    const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params) {
  auto isec = intersect_bvh(bvh, scene, ray);
  if (!isec.hit) return zero4f;
  auto& color = scene.materials[isec.instance].color;
  return vec4f{color.x, color.y, color.z, 0};
}

// Trace a single ray from the camera using the given algorithm.
using raytrace_shader_func = vec4f (*)(const scene_data& scene,
    const bvh_scene& bvh, const ray3f& ray, int bounce, rng_state& rng,
    const raytrace_params& params);
static raytrace_shader_func get_shader(const raytrace_params& params) {
  switch (params.shader) {
    case raytrace_shader_type::raytrace: return shade_raytrace;
    case raytrace_shader_type::matte: return shade_matte;
    case raytrace_shader_type::eyelight: return shade_eyelight;
    case raytrace_shader_type::normal: return shade_normal;
    case raytrace_shader_type::texcoord: return shade_texcoord;
    case raytrace_shader_type::color: return shade_color;
    case raytrace_shader_type::cell: return cell_shading;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}

// Build the bvh acceleration structure.
bvh_scene make_bvh(const scene_data& scene, const raytrace_params& params) {
  return make_bvh(scene, false, false, params.noparallel);
}

// Init a sequence of random number generators.
raytrace_state make_state(
    const scene_data& scene, const raytrace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = raytrace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image.assign(state.width * state.height, {0, 0, 0, 0});
  state.hits.assign(state.width * state.height, 0);
  state.rngs.assign(state.width * state.height, {});
  auto rng_ = make_rng(1301081);
  for (auto& rng : state.rngs) {
    rng = make_rng(961748941ull, rand1i(rng_, 1 << 31) / 2 + 1);
  }
  return state;
}

// Progressively compute an image by calling trace_samples multiple times.
void raytrace_samples(raytrace_state& state, const scene_data& scene,
    const bvh_scene& bvh, const raytrace_params& params) {
  if (state.samples >= params.samples) return;
  auto& camera = scene.cameras[params.camera];
  auto  shader = get_shader(params);
  state.samples += 1;
  if (params.samples == 1) {
    for (auto idx = 0; idx < state.width * state.height; idx++) {
      auto i = idx % state.width, j = idx / state.width;
      auto u = (i + 0.5f) / state.width, v = (j + 0.5f) / state.height;
      auto ray      = eval_camera(camera, {u, v});
      auto radiance = shader(scene, bvh, ray, 0, state.rngs[idx], params);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    }
  } else if (params.noparallel) {
    for (auto idx = 0; idx < state.width * state.height; idx++) {
      auto i = idx % state.width, j = idx / state.width;
      auto u        = (i + rand1f(state.rngs[idx])) / state.width,
           v        = (j + rand1f(state.rngs[idx])) / state.height;
      auto ray      = eval_camera(camera, {u, v});
      auto radiance = shader(scene, bvh, ray, 0, state.rngs[idx], params);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    }
  } else {
    parallel_for(state.width * state.height, [&](int idx) {
      auto i = idx % state.width, j = idx / state.width;
      auto u        = (i + rand1f(state.rngs[idx])) / state.width,
           v        = (j + rand1f(state.rngs[idx])) / state.height;
      auto ray      = eval_camera(camera, {u, v});
      auto radiance = shader(scene, bvh, ray, 0, state.rngs[idx], params);
      if (!isfinite(radiance)) radiance = {0, 0, 0};
      state.image[idx] += radiance;
      state.hits[idx] += 1;
    });
  }
}

// Check image type
static void check_image(
    const color_image& image, int width, int height, bool linear) {
  if (image.width != width || image.height != height)
    throw std::invalid_argument{"image should have the same size"};
  if (image.linear != linear)
    throw std::invalid_argument{
        linear ? "expected linear image" : "expected srgb image"};
}

// Get resulting render
color_image get_render(const raytrace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_render(image, state);
  return image;
}
void get_render(color_image& image, const raytrace_state& state) {
  check_image(image, state.width, state.height, true);
  auto scale = 1.0f / (float)state.samples;
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = state.image[idx] * scale;
  }
}

}  // namespace yocto

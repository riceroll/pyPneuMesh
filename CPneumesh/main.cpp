#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <chrono>

#include <imgui/imgui.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>

#include "Eigen/Core"

#include "./Model.h"

using namespace std;


int main(int argc, char **argv) {
  // initialize variables
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiMenu menu;

  auto* model = new Model();

//  auto begin = std::chrono::high_resolution_clock::now();

//  for (int i=0; i<100; i++) {
//    model->step();
//  }

//  auto end = std::chrono::high_resolution_clock::now();
//  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
//  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);


  // callback
  viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer& viewer)->bool {

//    model->step();

    viewer.data().clear_labels();
    viewer.data().clear();
    viewer.data().add_edges(model->V(model->E(Eigen::all, 0), Eigen::all),
                            model->V(model->E(Eigen::all, 1), Eigen::all),
                            Eigen::RowVector3d(0.5, 0.5, 0.5));


//    model->step(model->steps_per_frame);
//    viewer.data().set_mesh(model->V,model->F);
//    viewer.data().compute_normals();

    return true;
  };


  // menu
  menu.callback_draw_viewer_window = [&]()
  {
    // remove the default window
    viewer.data().clear_labels();
  };


  menu.callback_draw_custom_window =[&]()
  {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    ImGui::SetNextWindowPos(ImVec2(1.0 * mode->width / 2 - 150, 0), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(150, 0), ImGuiSetCond_FirstUseEver);

    // archived load function
    {
//    if (ImGui::Button("Load"))
//    {
//      std::string input_dir = igl::file_dialog_open();
//      string root_dir = ROOT_DIR;
//      root_dir += '/';
//      Pre processor p = Preprocessor(input_dir, root_dir);
//      igl::readOBJ(root_dir + "data/out.obj", V, F);
//      model = new Model(V, F, &viewer);
//      viewer.data().clear();
//      viewer.data().set_mesh(model->V, model->F);
//      C = V * 0;
//      for (int i=0; i<C.rows(); i++) {
//        C(i,0) = 0.6;
//        C(i,1) = 0.6;
//        C(i,2) = 0.8;
//      }
//      viewer.data().set_colors(C);
//    }
    }

    ImGui::Begin(
      "Setting", nullptr,
      ImGuiWindowFlags_NoSavedSettings
    );

    if (ImGui::Button("step")) {
      for (int i=0; i<model->nSteps; i++) {
        model->step();
      }
      printf("%.16f  %.16f  %.16f\n", model->V(0, 0), model->V(0, 1), model->V(0,2));
    }

    ImGui::InputDouble("h", &model->h, 0.0005, 0.01, "%.3g");
    ImGui::InputDouble("k", &model->k, 1000, 100000, "%.0g");
    ImGui::InputDouble("damping", &model->damping, 0.01, 0.05, "%.3g");

    ImGui::InputDouble("l0Ratio", &model->l0Ratio, 0.1, 0.2, "%.3g");
    ImGui::InputInt("nSteps", &model->nSteps, 100, 1000);


//    ImGui::InputDouble("tensile factor", &model->k_s, 0.001, 0.01, "%.3g");
//    ImGui::InputDouble("dielectric factor", &model->param->k_e, 0.001, 0.01, "%.3g");
//    ImGui::InputDouble("bending factor", &model->param->k_b, 0.001, 0.01, "%.3g");
//    ImGui::InputDouble("damping", &model->damping, 0.001, 0.01, "%.3g");
////    ImGui::InputDouble("damping factor", &model->damping_coeff, 0.001, 0.01, "%.3g");
//    ImGui::InputDouble("step size", &model->h);
//    ImGui::InputDouble("platform_on", &param->w_platform);
//    ImGui::InputInt("# steps/frame", &model->steps_per_frame);
////    ImGui::InputFloat("rad per frame", &model->rad_per_frame, 0.01, 0.1, "%.2g");
//    if (ImGui::Button("begin")) {
//      model->paused = false;
//    }
//    if (ImGui::Button("paused")) {
//      model->paused = true;
//    }
//    if (ImGui::Button("Reset")) {
//      loadModel(viewer, V, F, NF, model, ivs_fixed, param);
//    }
//    if (ImGui::Button("add anchor")) {
//      loadModel(viewer, V, F, NF, model, ivs_fixed, param);
//    }
//    if (ImGui::Button("Download")) {
//      char buff[100];
//      sprintf(buff, "%s/data/download/output.obj", ROOT_DIR);
//      string output_dir = buff;
//      igl::writeOBJ(output_dir, model->V, model->F);
//    }

//    static int num_choices = 0;
//    if (ImGui::InputInt("Num letters", &num_choices))
//    {
//      model->show_bending_force_i = false;
//      model->show_bending_force_j = false;
//      model->show_bending_force_k = false;
//      model->show_bending_force_l = false;
//      model->show_edge_force = false;
//      model->show_electrostatic_force = false;
//
//
//      switch(num_choices) {
//
//        case 0 :
//          model->show_bending_force_i = true;
//          break;
//
//        case 1 :
//          model->show_bending_force_j = true;
//          break;
//
//        case 2 :
//          model->show_bending_force_k = true;
//          break;
//
//        case 3 :
//          model->show_bending_force_l = true;
//          break;
//
//        case 4 :
//          model->show_electrostatic_force = true;
//          break;
//
//        case 5 :
//          model->show_edge_force = true;
//          break;
//
//        default :
//          cout<<"default"<<endl;
//          break;
//
//      }
//
//    }
    ImGui::End();
  };

  // viewer settings
  viewer.callback_init = [&](igl::opengl::glfw::Viewer& viewer) -> bool {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    glfwSetWindowPos(viewer.window, mode->width, 0);
    glfwSetWindowSize(viewer.window, mode->width, mode->height);
    return false;
  };

  viewer.plugins.push_back(&menu);
  viewer.core().is_animating = true;
  viewer.core().background_color = Eigen::Vector4f(0.9, 0.9, 0.9, 1.0);

  viewer.core().camera_zoom = 0.2;

  viewer.launch(true, false, "PneuMesh");
}
